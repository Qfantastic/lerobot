#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Continual Learning (CL) training of X-VLA on LIBERO benchmark.

Supported CL methods (--cl_method):

  sequential (default)
      Classic sequential CL: train task 0 → save model → load → train task 1 → ...
      Each stage inherits the previous task's weights. Exhibits catastrophic forgetting.
      Optional: combine with --use_lora for LoRA-based fine-tuning.

  tail
      TAIL: Task-specific Adapters for Imitation Learning.
      Each task trains a *separate* LoRA adapter on top of the *frozen base model*.
      Adapters are saved to a per-task library; inference selects the right adapter
      by task ID. Requires --use_lora.
      Reference: "TAIL: Task-specific Adapters for Imitation Learning with Large
      Pretrained Models" (arXiv:2410.11745)

LoRA library layout (TAIL mode):
  output_dir/lora_library/
    manifest.json                         <- task registry
    task_tidx0/adapter_model.safetensors  <- adapter weights (~MBs)
    task_tidx0/policy_preprocessor.json   <- per-task processor config
    task_tidx1/...

Key XVLA-specific notes vs Pi05:
  - Normalization: IDENTITY (no quantile stats required unlike Pi05/quantile error)
  - Tokenizer: facebook/bart-large (non-gated, no HF license gate needed)
  - VLM backbone: Florence2 (not PaliGemma). VLM uses 1/10 LR in full fine-tuning
    mode (XVLAAdamWConfig default), replicated here via --vlm_lr_scale.
  - Soft prompts: 32-dim learnable per-domain embeddings (30 domain slots).
    freeze flags: --freeze_vision_encoder, --freeze_language_encoder
    (replaces --train_expert_only from Pi05)
  - Default chunk_size=32 (not 50 as in Pi05)
  - Default action_mode=ee6d for LIBERO (end-effector + 6D rotation)

Usage examples:

  # TAIL with LoRA (recommended for CL research):
  python train_xvla_cl_libero.py \\
      --pretrained lerobot/xvla-base \\
      --dataset_repo_id lerobot/libero_10 \\
      --output_dir outputs/tail_xvla_libero10 \\
      --cl_method tail --use_lora \\
      --lora_r 64 --lora_alpha 64 \\
      --steps_per_task 10000

  # Sequential with LoRA + language encoder frozen (recommended):
  python train_xvla_cl_libero.py \\
      --dataset_repo_id lerobot/libero_10 \\
      --cl_method sequential --use_lora \\
      --lora_r 64 --lora_alpha 64 \\
      --freeze_language_encoder \\
      --steps_per_task 10000

  # Sequential full fine-tuning (official recommended: VLM 1/10 LR, transformer full LR):
  python train_xvla_cl_libero.py \\
      --dataset_repo_id lerobot/libero_10 \\
      --cl_method sequential \\
      --steps_per_task 10000
  # All params trainable: VLM at --vlm_lr_scale=0.1x, transformer at 1x, soft prompts at 1x
  # Matches XVLAAdamWConfig.build() differential-LR strategy.

  # Phase II (freeze VLM entirely, train only policy transformer + soft prompts):
  python train_xvla_cl_libero.py \\
      --dataset_repo_id lerobot/libero_10 \\
      --cl_method sequential \\
      --phase2 \\
      --steps_per_task 10000
  # 311M / 879M trainable (35%). Recommended when VLM features are already well aligned.

  # Sequential with LoRA + policy transformer (language encoder LoRA + transformer full):
  python train_xvla_cl_libero.py \\
      --dataset_repo_id lerobot/libero_10 \\
      --cl_method sequential --use_lora \\
      --lora_r 64 --lora_alpha 64 \\
      --steps_per_task 10000
  # LoRA on VLM Q/V projections; policy transformer + soft prompts also unfrozen.

  # Evaluate TAIL library with rollouts (requires libero installed):
  python train_xvla_cl_libero.py \\
      --output_dir outputs/tail_xvla_libero10 \\
      --eval_rollout --n_eval_episodes 20 --resume

  # Resume interrupted training:
  python train_xvla_cl_libero.py \\
      --output_dir outputs/tail_xvla_libero10 \\
      --resume

Multi-GPU examples (DDP via torchrun):

  # 4-GPU DDP (TAIL):
  torchrun --nproc_per_node=4 train_xvla_cl_libero.py --cl_method tail --use_lora ...

  # 2-GPU DDP (sequential):
  torchrun --nproc_per_node=2 train_xvla_cl_libero.py --cl_method sequential ...

  # Single GPU:
  python train_xvla_cl_libero.py --gpus 3 --cl_method tail --use_lora ...
"""

import argparse
import copy
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.xvla import XVLAConfig, XVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import cycle, init_logging


log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# LoRA utilities
# ══════════════════════════════════════════════════════════════════

def _require_peft():
    try:
        import peft  # noqa: F401
    except ImportError:
        raise ImportError(
            "peft is required for LoRA training. Install with:\n"
            "  pip install 'lerobot[peft-dep]'  or  pip install peft"
        )


class _XVLALoRABridge(torch.nn.Module):
    """Bridge between PeftModel and XVLAPolicy.

    PeftModelForFeatureExtraction.forward binds the first positional arg to
    `input_ids`. We pass the entire batch dict as that first positional arg,
    then route it to XVLAPolicy.forward(batch). The real `input_ids` key
    (tokenized language) inside the batch dict is handled by XVLAPolicy internally.
    """

    def __init__(self, policy: XVLAPolicy):
        super().__init__()
        self.policy = policy
        self.config = policy.config

    def forward(self, input_ids=None, **kwargs):
        if isinstance(input_ids, dict):
            return self.policy(input_ids)
        raise TypeError(
            f"_XVLALoRABridge.forward: expected batch dict in 'input_ids', "
            f"got {type(input_ids)}."
        )


def _parse_lora_targets(s: str):
    """Return regex string or list[str] for LoraConfig.target_modules."""
    if any(c in s for c in ('|', '(', '[', '\\')):
        return s
    return [m.strip() for m in s.split(",") if m.strip()]


def apply_lora(
    policy: XVLAPolicy,
    r: int,
    alpha: int,
    dropout: float,
    target_modules,
    train_policy_transformer: bool = True,
    train_soft_prompts: bool = True,
) -> Any:
    """Wrap policy in a PEFT LoRA model; return PeftModel for training.

    PEFT's get_peft_model() freezes ALL base model params and adds trainable LoRA
    adapters. For sequential CL (and any mode where we want the policy transformer
    to adapt), we re-enable requires_grad on the transformer and soft prompts
    *after* PEFT setup, so they train alongside the LoRA adapters.

    For TAIL (per-task adapters on a frozen base), pass train_policy_transformer=False
    and train_soft_prompts=False to keep only LoRA adapters trainable.
    """
    _require_peft()
    from peft import LoraConfig, TaskType, get_peft_model

    bridge = _XVLALoRABridge(policy)
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(bridge, lora_cfg)

    # Re-enable policy transformer and/or soft prompts after PEFT froze everything.
    base_policy = peft_model.base_model.model.policy
    if train_policy_transformer:
        for name, param in base_policy.model.transformer.named_parameters():
            if "soft_prompt" not in name:
                param.requires_grad = True
    if train_soft_prompts and hasattr(base_policy.model.transformer, "soft_prompt_hub"):
        base_policy.model.transformer.soft_prompt_hub.weight.requires_grad = True

    peft_model.print_trainable_parameters()
    return peft_model


def save_lora_adapter(peft_model: Any, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(save_dir))
    log.info(f"LoRA adapter saved -> {save_dir}")


def load_lora_adapter(base_policy: XVLAPolicy, adapter_dir: Path) -> Any:
    _require_peft()
    from peft import PeftModel

    bridge = _XVLALoRABridge(base_policy)
    return PeftModel.from_pretrained(bridge, str(adapter_dir), is_trainable=False)


def merge_lora_into_policy(peft_model: Any) -> XVLAPolicy:
    """Merge LoRA weights into the base XVLAPolicy and return it."""
    merged_bridge: _XVLALoRABridge = peft_model.merge_and_unload()
    return merged_bridge.policy


def get_base_config(policy: Any) -> XVLAConfig:
    p = unwrap_dp(policy)
    if hasattr(p, "base_model"):
        return p.base_model.model.config
    if isinstance(p, _XVLALoRABridge):
        return p.config
    return p.config


def get_trainable_params(policy: Any) -> list:
    return [p for p in policy.parameters() if p.requires_grad]


# ══════════════════════════════════════════════════════════════════
# Multi-GPU utilities
# ══════════════════════════════════════════════════════════════════

def parse_gpus(gpus_str: str) -> list[int]:
    return [int(g.strip()) for g in gpus_str.split(",") if g.strip()]


def primary_device(gpu_ids: list[int]) -> torch.device:
    if gpu_ids and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_ids[0]}")
    return torch.device("cpu")


def init_ddp() -> tuple[bool, int, int, int]:
    if "RANK" not in os.environ:
        return False, 0, 0, 1
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return True, local_rank, rank, world_size


def wrap_dp(
    policy: Any,
    gpu_ids: list[int],
    local_rank: int = 0,
    is_ddp: bool = False,
) -> Any:
    if is_ddp:
        return DDP(policy, device_ids=[local_rank], find_unused_parameters=False)
    if len(gpu_ids) > 1:
        log.warning(
            f"  --gpus {gpu_ids}: DataParallel disabled. "
            f"Use `torchrun --nproc_per_node={len(gpu_ids)}` for multi-GPU DDP."
        )
    return policy


def unwrap_dp(policy: Any) -> Any:
    if isinstance(policy, (torch.nn.DataParallel, DDP)):
        return policy.module
    return policy


def log_gpu_stats(gpu_ids: list[int], prefix: str = "") -> None:
    if not torch.cuda.is_available():
        return
    parts = []
    for gid in gpu_ids:
        alloc = torch.cuda.memory_allocated(gid) / 1024**3
        reserved = torch.cuda.memory_reserved(gid) / 1024**3
        parts.append(f"GPU{gid}: {alloc:.1f}/{reserved:.1f} GB")
    log.info(f"  {prefix}GPU mem (alloc/reserved): {' | '.join(parts)}")


# ══════════════════════════════════════════════════════════════════
# LoRA Library (TAIL)
# ══════════════════════════════════════════════════════════════════

class LoRALibrary:
    """Manages per-task LoRA adapters for the TAIL method."""

    def __init__(self, library_dir: Path):
        self.library_dir = library_dir
        self.manifest_path = library_dir / "manifest.json"
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {"base_model": "", "dataset_repo_id": "", "adapters": {}}

    def save(self) -> None:
        self.library_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self._data, f, indent=2, default=str)

    def set_meta(self, base_model: str, dataset_repo_id: str) -> None:
        self._data["base_model"] = base_model
        self._data["dataset_repo_id"] = dataset_repo_id
        self.save()

    def register(
        self,
        task_index: int,
        task_name: str,
        adapter_path: Path,
        preprocessor_path: Path,
    ) -> None:
        self._data["adapters"][str(task_index)] = {
            "task_index": task_index,
            "task_name": task_name,
            "adapter_path": str(adapter_path),
            "preprocessor_path": str(preprocessor_path),
        }
        self.save()

    def get(self, task_index: int) -> dict | None:
        return self._data["adapters"].get(str(task_index))

    @property
    def base_model(self) -> str:
        return self._data["base_model"]

    @property
    def all_adapters(self) -> dict:
        return self._data["adapters"]

    def has_task(self, task_index: int) -> bool:
        return str(task_index) in self._data["adapters"]

    def __repr__(self) -> str:
        n = len(self._data["adapters"])
        return f"LoRALibrary({self.library_dir}, {n} adapters)"


# ══════════════════════════════════════════════════════════════════
# Suite -> task index mapping
# ══════════════════════════════════════════════════════════════════

SUITE_TASK_RANGES: dict[str, range] = {
    "libero_spatial": range(0, 10),
    "libero_object":  range(10, 20),
    "libero_goal":    range(20, 30),
    "libero_10":      range(30, 40),
}

SUITE_ALIASES: dict[str, str] = {
    "spatial":        "libero_spatial",
    "object":         "libero_object",
    "goal":           "libero_goal",
    "libero_goal":    "libero_goal",
    "libero_spatial": "libero_spatial",
    "libero_object":  "libero_object",
    "long":           "libero_10",
    "libero_long":    "libero_10",
    "libero_10":      "libero_10",
}


def resolve_suite(suite_name: str) -> list[int]:
    key = SUITE_ALIASES.get(suite_name.lower(), suite_name.lower())
    if key not in SUITE_TASK_RANGES:
        valid = list(SUITE_ALIASES.keys())
        raise ValueError(f"Unknown suite {suite_name!r}. Valid options: {valid}")
    return list(SUITE_TASK_RANGES[key])


# ══════════════════════════════════════════════════════════════════
# Policy loading helpers
# ══════════════════════════════════════════════════════════════════

def build_policy_features_from_dataset(
    ds_meta: LeRobotDatasetMetadata,
) -> tuple[dict, dict]:
    """Derive PolicyFeature dicts (input, output) from a LeRobotDatasetMetadata."""
    from lerobot.configs.types import FeatureType, PolicyFeature

    _SKIP = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    input_features: dict = {}
    output_features: dict = {}

    for feat_key, feat_info in ds_meta.features.items():
        if feat_key in _SKIP:
            continue
        dtype = feat_info.get("dtype", "")
        shape = feat_info.get("shape", [])

        if dtype in ("video", "image"):
            if len(shape) == 3 and shape[2] in (1, 3, 4):
                policy_shape = (shape[2], shape[0], shape[1])
            else:
                policy_shape = tuple(shape)
            input_features[feat_key] = PolicyFeature(type=FeatureType.VISUAL, shape=policy_shape)
        elif feat_key == "observation.state":
            input_features[feat_key] = PolicyFeature(type=FeatureType.STATE, shape=tuple(shape))
        elif feat_key == "action":
            output_features[feat_key] = PolicyFeature(type=FeatureType.ACTION, shape=tuple(shape))

    return input_features, output_features


def cache_pretrained_weights(pretrained: str) -> tuple[Any, dict]:
    """Load pretrained XVLA once and return (config, cpu_state_dict).

    Used by run_tail to avoid reloading ~0.9B Florence2 weights N times.
    CLI overrides are applied later in load_base_policy() per-task.
    """
    log.info(f"  Pre-loading pretrained XVLA weights (reused for all tasks): {pretrained}")
    base = XVLAPolicy.from_pretrained(pretrained)
    config = base.config
    state = {k: v.cpu() for k, v in base.state_dict().items()}
    del base
    torch.cuda.empty_cache()
    return config, state


def load_base_policy(
    pretrained: str,
    ds_meta: LeRobotDatasetMetadata,
    device: torch.device,
    args: argparse.Namespace | None = None,
    _cached: tuple | None = None,
) -> XVLAPolicy:
    """Load XVLAPolicy with features adapted to the dataset.

    XVLA notes vs Pi05:
    - XVLAPolicy(config) initializes Florence2 from config (random weights).
      load_state_dict(strict=False) loads the cached pretrained weights.
    - IDENTITY normalization means dataset_stats are not needed for normalizer.
    - Tokenizer is facebook/bart-large (non-gated).
    - CLI flags override config: dtype, freeze_vision_encoder,
      freeze_language_encoder, train_policy_transformer, train_soft_prompts,
      action_mode, num_denoising_steps.
    - In full fine-tuning mode, VLM params use vlm_lr_scale * lr (see
      _make_optimizer_and_scheduler), matching XVLAAdamWConfig's 1/10 VLM LR.
    """
    input_features, output_features = build_policy_features_from_dataset(ds_meta)
    log.info(f"  Dataset input_features : {list(input_features.keys())}")
    log.info(f"  Dataset output_features: {list(output_features.keys())}")

    if _cached is not None:
        config, base_state = _cached
        config = copy.copy(config)
    else:
        log.info(f"  Loading pretrained XVLA: {pretrained}")
        base = XVLAPolicy.from_pretrained(pretrained)
        config = base.config
        base_state = {k: v.cpu() for k, v in base.state_dict().items()}
        del base
        if device.type == "cuda":
            torch.cuda.empty_cache()

    config.input_features = input_features
    config.output_features = output_features

    # Apply CLI overrides for XVLA-specific training flags
    if args is not None:
        config.dtype = args.dtype
        config.freeze_vision_encoder = args.freeze_vision_encoder
        config.freeze_language_encoder = args.freeze_language_encoder
        config.train_policy_transformer = args.train_policy_transformer
        config.train_soft_prompts = args.train_soft_prompts
        config.action_mode = args.action_mode
        config.num_denoising_steps = args.num_denoising_steps
        config.chunk_size = args.chunk_size
        config.n_obs_steps = args.n_obs_steps
        config.tokenizer_max_length = args.tokenizer_max_length
        log.info(f"  dtype={config.dtype}  freeze_vision={config.freeze_vision_encoder}  "
                 f"freeze_language={config.freeze_language_encoder}  "
                 f"train_transformer={config.train_policy_transformer}  "
                 f"train_soft_prompts={config.train_soft_prompts}  "
                 f"action_mode={config.action_mode}")

    # The pretrained model has tokenizer_max_length=1024 + pad_language_to=max_length.
    # LIBERO task descriptions are ~25-35 tokens; padding to 1024 creates vlm_features
    # of ~1073 tokens which exceeds the policy transformer's max_len_seq=512.
    # Override to pad only to longest in each batch.
    config.pad_language_to = "longest"

    policy = XVLAPolicy(config)
    missing, unexpected = policy.load_state_dict(base_state, strict=False)
    if missing:
        log.info(f"  Randomly initialized (dim mismatch): {len(missing)} keys "
                 f"(e.g. {missing[:3]})")
    if unexpected:
        log.info(f"  Skipped (not in new model): {len(unexpected)} keys")

    # Re-apply dtype after load_state_dict: the cached state may be in a different
    # dtype from the pretrained model config. This mirrors XVLAPolicy.from_pretrained
    # which calls instance.model._apply_dtype() after loading weights.
    policy.model._apply_dtype()

    return policy.to(device)


# ══════════════════════════════════════════════════════════════════
# Dataset helpers
# ══════════════════════════════════════════════════════════════════

def get_task_name(ds_meta: LeRobotDatasetMetadata, task_index: int) -> str:
    matches = ds_meta.tasks[ds_meta.tasks["task_index"] == task_index]
    return matches.index[0] if len(matches) > 0 else f"task_{task_index}"


def build_lerobot_to_libero_task_id_map(suite_name: str, ds_meta: LeRobotDatasetMetadata) -> dict[int, int]:
    from libero.libero import benchmark as libero_benchmark

    bm = libero_benchmark.get_benchmark_dict()
    if suite_name not in bm:
        raise ValueError(f"Suite '{suite_name}' not found in LIBERO benchmark. Available: {list(bm.keys())}")
    suite = bm[suite_name]()

    libero_lang_to_id: dict[str, int] = {}
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        libero_lang_to_id[task.language.strip().lower()] = i

    mapping: dict[int, int] = {}
    for lerobot_name, row in ds_meta.tasks.iterrows():
        lerobot_idx = int(row["task_index"])
        norm = str(lerobot_name).strip().lower()
        libero_id = libero_lang_to_id.get(norm)
        if libero_id is None:
            raise ValueError(
                f"Cannot match LeRobot task_index={lerobot_idx} '{lerobot_name}' "
                f"to any LIBERO task in suite '{suite_name}'.\n"
                f"Available LIBERO tasks: {list(libero_lang_to_id.keys())}"
            )
        mapping[lerobot_idx] = libero_id

    log.info("  LeRobot->LIBERO task_id mapping:")
    for lr_idx, lib_id in sorted(mapping.items()):
        log.info(f"    LeRobot task_index={lr_idx} -> LIBERO task_id={lib_id}")
    return mapping


def build_delta_timestamps(
    ds_meta: LeRobotDatasetMetadata,
    chunk_size: int,
    n_obs_steps: int,
) -> dict[str, list[float]]:
    fps = ds_meta.fps
    obs_ts = [i / fps for i in range(n_obs_steps)]
    action_ts = [i / fps for i in range(chunk_size)]
    delta_ts: dict[str, list[float]] = {}
    for key in ds_meta.camera_keys:
        delta_ts[key] = obs_ts
    if "observation.state" in ds_meta.features:
        delta_ts["observation.state"] = obs_ts
    delta_ts["action"] = action_ts
    return delta_ts


def build_full_dataset(
    repo_id: str,
    ds_meta: LeRobotDatasetMetadata,
    chunk_size: int,
    n_obs_steps: int,
    root: str | None,
) -> LeRobotDataset:
    delta_ts = build_delta_timestamps(ds_meta, chunk_size, n_obs_steps)
    log.info(f"  Loading full dataset {repo_id!r}...")
    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=None,
        delta_timestamps=delta_ts,
    )


class TaskFilterDataset(torch.utils.data.Dataset):
    """Wraps a full LeRobotDataset and exposes only frames for one task_index."""

    def __init__(self, full_dataset: LeRobotDataset, task_index: int):
        arrow = full_dataset.hf_dataset.data
        task_col = arrow.column("task_index").to_pylist()
        self._indices: list[int] = [i for i, t in enumerate(task_col) if t == task_index]
        self._full = full_dataset
        self.meta = full_dataset.meta
        self.num_frames = len(self._indices)
        log.info(f"  TaskFilterDataset: task_index={task_index}, frames={self.num_frames}")

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        return self._full[self._indices[idx]]


# ══════════════════════════════════════════════════════════════════
# LR schedule
# ══════════════════════════════════════════════════════════════════

def make_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    decay_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return decay_lr_ratio + (1.0 - decay_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ══════════════════════════════════════════════════════════════════
# Core training loop
# ══════════════════════════════════════════════════════════════════

def train_steps(
    policy: Any,
    dataset: torch.utils.data.Dataset,
    preprocessor: Any,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    steps: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    grad_clip_norm: float = 10.0,
    log_freq: int = 100,
    rank: int = 0,
    world_size: int = 1,
    wandb_log: "Any | None" = None,
    global_step_offset: int = 0,
) -> dict[str, float]:
    collate_fn = lerobot_collate_fn if dataset.meta.has_language_columns else None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=(num_workers > 0),
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            collate_fn=collate_fn,
            persistent_workers=(num_workers > 0),
        )

    if world_size > 1:
        _epoch = 0
        _sampler = sampler
        dl_iter = iter(dataloader)
    else:
        dl_iter = cycle(dataloader)
        _sampler = None

    policy.train()
    camera_keys = dataset.meta.camera_keys
    losses: list[float] = []
    t0 = time.perf_counter()

    pbar = tqdm(range(steps), desc="Training", unit="step", disable=(rank != 0))
    for step in pbar:
        if world_size > 1:
            try:
                batch = next(dl_iter)
            except StopIteration:
                _epoch += 1
                _sampler.set_epoch(_epoch)
                dl_iter = iter(dataloader)
                batch = next(dl_iter)
        else:
            batch = next(dl_iter)

        for k in camera_keys:
            if k in batch and isinstance(batch[k], torch.Tensor) and batch[k].dtype == torch.uint8:
                batch[k] = batch[k].to(dtype=torch.float32) / 255.0

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = preprocessor(batch)

        out = policy.forward(batch)
        loss = out[0] if isinstance(out, tuple) else out
        if not isinstance(loss, torch.Tensor):
            raise RuntimeError(f"Expected loss tensor from policy.forward(), got {type(loss)}")
        if loss.dim() > 0:
            loss = loss.mean()
        loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)

        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)

        if (step + 1) % log_freq == 0:
            window = losses[-log_freq:]
            avg = sum(window) / len(window)
            elapsed = time.perf_counter() - t0
            cur_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{avg:.4f}", "t": f"{elapsed:.0f}s"})
            log.info(f"  step={step+1}/{steps}  loss={avg:.4f}  lr={cur_lr:.2e}  elapsed={elapsed:.1f}s")
            if wandb_log is not None and rank == 0:
                wandb_log({
                    "train/loss": avg,
                    "train/lr": cur_lr,
                    "train/step": global_step_offset + step + 1,
                })

    return {"mean_loss": sum(losses) / len(losses), "final_loss": losses[-1]}


def _make_optimizer_and_scheduler(policy: Any, args: argparse.Namespace) -> tuple:
    """Build optimizer and LR scheduler for XVLA.

    Mirrors XVLAAdamWConfig.build() with three parameter groups:
      - vlm        : Florence2 params   → lr * vlm_lr_scale (default 0.1x)
      - soft_prompts: soft_prompt_hub   → lr * soft_prompt_lr_scale (default 1.0x)
      - transformer : policy head + rest→ lr (1.0x)

    In LoRA mode the same grouping applies: LoRA adapters land in "vlm" or
    "transformer" depending on their name, soft prompts stay separate.
    """
    soft_prompt_lr_scale = getattr(args, "soft_prompt_lr_scale", 1.0)

    vlm_params, soft_prompt_params, other_params = [], [], []
    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if "soft_prompt" in name:
            soft_prompt_params.append(param)
        elif "vlm" in name:
            vlm_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": args.lr, "name": "transformer"})
    if soft_prompt_params:
        param_groups.append({
            "params": soft_prompt_params,
            "lr": args.lr * soft_prompt_lr_scale,
            "name": "soft_prompts",
        })
    if vlm_params:
        param_groups.append({
            "params": vlm_params,
            "lr": args.lr * args.vlm_lr_scale,
            "name": "vlm",
        })
    if not param_groups:
        raise ValueError(
            "No trainable parameters found. Check freeze flags — "
            "--freeze_vision_encoder + --freeze_language_encoder + "
            "--no_train_policy_transformer leaves nothing to train."
        )
    for g in param_groups:
        log.info(f"  Optimizer group '{g['name']}': {len(g['params'])} tensors, lr={g['lr']:.2e}")
    optimizer = torch.optim.AdamW(
        param_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10
    )

    warmup = min(args.warmup_steps, args.steps_per_task // 10)
    lr_scheduler = make_cosine_warmup_scheduler(
        optimizer, warmup, args.steps_per_task, args.decay_lr_ratio
    )
    return optimizer, lr_scheduler


# ══════════════════════════════════════════════════════════════════
# CL Method: Sequential
# ══════════════════════════════════════════════════════════════════

def run_sequential(
    args: argparse.Namespace,
    ds_meta: LeRobotDatasetMetadata,
    task_indices: list[int],
    output_dir: Path,
    cl_state: dict,
    device: torch.device | None = None,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    is_main: bool = True,
    wandb_log=None,
) -> dict:
    gpu_ids = parse_gpus(args.gpus)
    if device is None:
        device = primary_device(gpu_ids)
    if is_main and gpu_ids:
        log.info(f"  GPUs: {gpu_ids}  local_rank: {local_rank}  world_size: {world_size}  device: {device}")
        log_gpu_stats([local_rank] if world_size > 1 else gpu_ids, "initial ")
    completed = {t["task_index"] for t in cl_state["tasks_completed"]}

    if cl_state["task_checkpoints"]:
        last = max(int(k) for k in cl_state["task_checkpoints"])
        current_path = cl_state["task_checkpoints"][str(last)]
    else:
        current_path = args.pretrained

    full_dataset = build_full_dataset(
        args.dataset_repo_id, ds_meta, args.chunk_size, args.n_obs_steps, args.dataset_root
    )

    for task_order, task_idx in enumerate(task_indices):
        if task_idx in completed:
            log.info(f"[{task_order+1}/{len(task_indices)}] Skipping task {task_idx} (done).")
            continue

        task_name = get_task_name(ds_meta, task_idx)
        log.info(f"\n{'='*60}")
        log.info(f"[Sequential] {task_order+1}/{len(task_indices)} | task_index={task_idx}")
        log.info(f"  Task: {task_name!r}")
        log.info(f"  Loading from: {current_path}")

        task_dataset = TaskFilterDataset(full_dataset, task_idx)
        if task_dataset.num_frames == 0:
            log.warning(f"  No frames for task {task_idx}. Skipping.")
            continue
        log.info(f"  Frames: {task_dataset.num_frames}")

        policy = load_base_policy(current_path, ds_meta, device, args=args)

        if args.use_lora:
            lora_targets = _parse_lora_targets(args.lora_target_modules)
            policy = apply_lora(
                policy, args.lora_r, args.lora_alpha, args.lora_dropout, lora_targets,
                train_policy_transformer=args.train_policy_transformer,
                train_soft_prompts=args.train_soft_prompts,
            )

        policy = wrap_dp(policy, gpu_ids, local_rank=local_rank, is_ddp=(world_size > 1))

        base_cfg = get_base_config(policy)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=base_cfg, dataset_stats=task_dataset.meta.stats
        )

        n_train = sum(p.numel() for p in get_trainable_params(policy))
        n_total = sum(p.numel() for p in policy.parameters())
        log.info(f"  Trainable: {n_train:,} / {n_total:,}")

        optimizer, lr_scheduler = _make_optimizer_and_scheduler(policy, args)

        train_stats = train_steps(
            policy, task_dataset, preprocessor, optimizer, lr_scheduler,
            args.steps_per_task, device, args.batch_size, args.num_workers,
            args.grad_clip_norm, args.log_freq,
            rank=rank, world_size=world_size,
            wandb_log=wandb_log,
            global_step_offset=task_order * args.steps_per_task,
        )

        if world_size > 1:
            dist.barrier()

        if is_main:
            ckpt_dir = output_dir / f"task_{task_order:02d}_tidx{task_idx}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            if args.use_lora:
                merged_policy = merge_lora_into_policy(unwrap_dp(policy))
                merged_policy.save_pretrained(ckpt_dir)
                preprocessor.save_pretrained(ckpt_dir)
                postprocessor.save_pretrained(ckpt_dir)
            else:
                unwrap_dp(policy).save_pretrained(ckpt_dir)
                preprocessor.save_pretrained(ckpt_dir)
                postprocessor.save_pretrained(ckpt_dir)

            current_path = str(ckpt_dir)
            _update_cl_state(cl_state, task_order, task_idx, task_name, task_dataset,
                             current_path, train_stats, output_dir)
            log.info(f"  Checkpoint -> {ckpt_dir}")
        else:
            current_path = str(output_dir / f"task_{task_order:02d}_tidx{task_idx}")

        if world_size > 1:
            dist.barrier()

        del policy, preprocessor, postprocessor, optimizer, lr_scheduler, task_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return cl_state


# ══════════════════════════════════════════════════════════════════
# CL Method: TAIL
# ══════════════════════════════════════════════════════════════════

def run_tail(
    args: argparse.Namespace,
    ds_meta: LeRobotDatasetMetadata,
    task_indices: list[int],
    output_dir: Path,
    cl_state: dict,
    device: torch.device | None = None,
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
    is_main: bool = True,
    wandb_log=None,
) -> tuple[dict, LoRALibrary]:
    """TAIL: per-task LoRA adapter, always trained on frozen base model."""
    gpu_ids = parse_gpus(args.gpus)
    if device is None:
        device = primary_device(gpu_ids)
    if is_main and gpu_ids:
        log.info(f"  GPUs: {gpu_ids}  local_rank: {local_rank}  world_size: {world_size}  device: {device}")
        log_gpu_stats([local_rank] if world_size > 1 else gpu_ids, "initial ")

    full_dataset = build_full_dataset(
        args.dataset_repo_id, ds_meta, args.chunk_size, args.n_obs_steps, args.dataset_root
    )

    library_dir = output_dir / "lora_library"
    lora_lib = LoRALibrary(library_dir)
    if is_main:
        lora_lib.set_meta(args.pretrained, args.dataset_repo_id)
    if world_size > 1:
        dist.barrier()

    pretrained_cache = cache_pretrained_weights(args.pretrained)
    completed = {t["task_index"] for t in cl_state["tasks_completed"]}

    for task_order, task_idx in enumerate(task_indices):
        if lora_lib.has_task(task_idx) and task_idx in completed:
            log.info(f"[{task_order+1}/{len(task_indices)}] Skipping task {task_idx} (done).")
            continue

        task_name = get_task_name(ds_meta, task_idx)
        log.info(f"\n{'='*60}")
        log.info(f"[TAIL] {task_order+1}/{len(task_indices)} | task_index={task_idx}")
        log.info(f"  Task: {task_name!r}")
        log.info(f"  Base model: {args.pretrained}  (TAIL always resets to base)")

        task_dataset = TaskFilterDataset(full_dataset, task_idx)
        if task_dataset.num_frames == 0:
            log.warning(f"  No frames for task {task_idx}. Skipping.")
            continue
        log.info(f"  Frames: {task_dataset.num_frames}")

        base_policy = load_base_policy(args.pretrained, ds_meta, device, args=args, _cached=pretrained_cache)

        lora_targets = _parse_lora_targets(args.lora_target_modules)
        # TAIL: base model stays fully frozen; only per-task LoRA adapters train.
        policy = apply_lora(
            base_policy, args.lora_r, args.lora_alpha, args.lora_dropout, lora_targets,
            train_policy_transformer=False,
            train_soft_prompts=False,
        )

        policy = wrap_dp(policy, gpu_ids, local_rank=local_rank, is_ddp=(world_size > 1))

        base_cfg = get_base_config(policy)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=base_cfg, dataset_stats=task_dataset.meta.stats
        )

        n_train = sum(p.numel() for p in get_trainable_params(policy))
        n_total = sum(p.numel() for p in policy.parameters())
        if is_main:
            log.info(f"  LoRA trainable: {n_train:,} / {n_total:,}")

        optimizer, lr_scheduler = _make_optimizer_and_scheduler(policy, args)

        train_stats = train_steps(
            policy, task_dataset, preprocessor, optimizer, lr_scheduler,
            args.steps_per_task, device, args.batch_size, args.num_workers,
            args.grad_clip_norm, args.log_freq,
            rank=rank, world_size=world_size,
            wandb_log=wandb_log,
            global_step_offset=task_order * args.steps_per_task,
        )

        if world_size > 1:
            dist.barrier()

        if is_main:
            adapter_dir = library_dir / f"task_tidx{task_idx}"
            save_lora_adapter(unwrap_dp(policy), adapter_dir)
            preprocessor.save_pretrained(adapter_dir)
            postprocessor.save_pretrained(adapter_dir)
            lora_lib.register(task_idx, task_name, adapter_dir, adapter_dir)
            _update_cl_state(cl_state, task_order, task_idx, task_name, task_dataset,
                             str(adapter_dir), train_stats, output_dir)
            log.info(f"  LoRA adapter -> {adapter_dir}")
            log.info(f"  Library: {lora_lib}")

        del policy, base_policy, preprocessor, postprocessor, optimizer, lr_scheduler, task_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return cl_state, lora_lib


def _update_cl_state(
    cl_state: dict,
    task_order: int,
    task_idx: int,
    task_name: str,
    task_dataset: torch.utils.data.Dataset,
    checkpoint_path: str,
    train_stats: dict,
    output_dir: Path,
) -> None:
    num_frames = getattr(task_dataset, "num_frames", len(task_dataset))
    cl_state["tasks_completed"].append({
        "order": task_order,
        "task_index": task_idx,
        "task_name": task_name,
        "num_frames": num_frames,
    })
    cl_state["task_checkpoints"][str(task_order)] = checkpoint_path
    cl_state["task_stats"][str(task_order)] = {
        **train_stats, "task_index": task_idx, "task_name": task_name,
    }
    state_path = output_dir / "cl_state.json"
    with open(state_path, "w") as f:
        json.dump(cl_state, f, indent=2, default=str)


# ══════════════════════════════════════════════════════════════════
# Rollout-based evaluation (requires libero installed)
# ══════════════════════════════════════════════════════════════════

def _run_rollout_episodes(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    suite_name: str,
    libero_task_id: int,
    n_episodes: int,
    device: torch.device,
    obs_height: int = 360,
    obs_width: int = 360,
    camera_name_mapping: dict | None = None,
    videos_dir: Path | None = None,
) -> dict[str, float]:
    from lerobot.envs.libero import create_libero_envs
    from lerobot.scripts.lerobot_eval import rollout
    import gymnasium as gym

    envs = create_libero_envs(
        task=suite_name,
        n_envs=1,
        gym_kwargs={
            "task_ids": [libero_task_id],
            "observation_height": obs_height,
            "observation_width": obs_width,
            "obs_type": "pixels_agent_pos",
        },
        camera_name_mapping=camera_name_mapping,
        env_cls=gym.vector.SyncVectorEnv,
        init_states=True,
    )
    vec_env = envs[suite_name][libero_task_id]
    render_fps = vec_env.unwrapped.metadata.get("render_fps", 20)

    policy.eval()
    successes = []

    for ep in range(n_episodes):
        ep_frames: list | None = [] if videos_dir is not None else None

        def _render_callback(env, _frames=ep_frames):
            if _frames is not None:
                _frames.append(env.envs[0].render())

        result = rollout(
            env=vec_env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            seeds=[ep],
            render_callback=_render_callback if videos_dir is not None else None,
        )
        ep_success = bool(result["success"].any())
        successes.append(ep_success)

        if videos_dir is not None and ep_frames:
            from lerobot.utils.io_utils import write_video
            videos_dir.mkdir(parents=True, exist_ok=True)
            outcome = "success" if ep_success else "fail"
            video_path = videos_dir / f"ep{ep + 1:02d}_{outcome}.mp4"
            write_video(str(video_path), ep_frames, render_fps)
            log.info(f"    episode {ep+1}/{n_episodes}: {'SUCCESS' if ep_success else 'fail'}  -> {video_path.name}")
        else:
            log.info(f"    episode {ep+1}/{n_episodes}: {'SUCCESS' if ep_success else 'fail'}")

    vec_env.close()

    success_rate = sum(successes) / len(successes) if successes else 0.0
    return {
        "success_rate": success_rate,
        "n_success": sum(successes),
        "n_episodes": n_episodes,
    }


def eval_sequential(
    args: argparse.Namespace,
    cl_state: dict,
    ds_meta: LeRobotDatasetMetadata,
    task_indices: list[int],
    output_dir: Path,
    device: torch.device,
    eval_output_dir: Path | None = None,
) -> dict:
    """Evaluate sequential CL: each task uses its own saved full checkpoint.

    If args.eval_task is set, ALL tasks are evaluated using that single checkpoint,
    enabling measurement of catastrophic forgetting across the CL sequence.
    """
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors

    results = {}
    suite_name = args.eval_suite or args.dataset_repo_id.split("/")[-1].replace("_image", "")

    fixed_ckpt_path = None
    if args.eval_task is not None:
        fixed_ckpt_key = str(args.eval_task)
        if fixed_ckpt_key not in cl_state["task_checkpoints"]:
            raise ValueError(
                f"--eval_task {args.eval_task} not found in saved checkpoints. "
                f"Available task_orders: {sorted(cl_state['task_checkpoints'].keys())}"
            )
        fixed_ckpt_path = cl_state["task_checkpoints"][fixed_ckpt_key]
        log.info(f"  [CL-forgetting mode] All tasks evaluated with checkpoint from "
                 f"task_order={args.eval_task}: {fixed_ckpt_path}")

    log.info("  Building LeRobot->LIBERO task_id mapping...")
    task_id_map = build_lerobot_to_libero_task_id_map(suite_name, ds_meta)

    for task_order, task_idx in enumerate(task_indices):
        ckpt_key = str(task_order)
        if ckpt_key not in cl_state["task_checkpoints"]:
            log.warning(f"No checkpoint for task_order={task_order} (task_idx={task_idx}). Skipping eval.")
            continue

        libero_task_id = task_id_map[task_idx]
        ckpt_path = fixed_ckpt_path if fixed_ckpt_path is not None else cl_state["task_checkpoints"][ckpt_key]
        task_name = get_task_name(ds_meta, task_idx)
        log.info(f"\n[Eval-Sequential] task_idx={task_idx} -> LIBERO task_id={libero_task_id} | {task_name!r}")
        log.info(f"  Loading: {ckpt_path}")

        policy = XVLAPolicy.from_pretrained(ckpt_path).to(device)
        policy.eval()

        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
        from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            ckpt_path, config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            ckpt_path, config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

        _cam_map = {"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}
        env_cfg = LiberoEnvConfig(task=suite_name, task_ids=[libero_task_id], camera_name_mapping=_cam_map)
        # env_preprocessor must only do LiberoProcessorStep (flip images + process state).
        # The policy preprocessor (loaded from checkpoint) already contains image_to_float,
        # imagenet_normalize, add_domain_id, and tokenize. Applying make_env_pre_post_processors
        # (which also includes imagenet_normalize + domain_id) would double-process images,
        # causing XVLAImageToFloatProcessorStep to fail on already-normalized values.
        from lerobot.policies.xvla.processor_xvla import LiberoProcessorStep
        from lerobot.processor import PolicyProcessorPipeline as _PPL
        env_preprocessor = _PPL(steps=[LiberoProcessorStep()])
        # env_postprocessor must be identity: XVLARotation6DToAxisAngleProcessorStep expects
        # 10D input [3pos+6rot+1grip] but AutoActionSpace.postprocess() already trims model
        # output (20D) to real_dim (7D axis-angle). Applying it would extract wrong slices
        # (gripper at [:, 9:10] from a 7-element tensor = empty) → 6D output instead of 7D.
        env_postprocessor = _PPL(steps=[])

        _vdir_root = eval_output_dir if eval_output_dir is not None else output_dir
        videos_dir = (_vdir_root / "eval_videos" / f"task{task_idx}") if args.save_videos else None
        metrics = _run_rollout_episodes(
            policy, preprocessor, postprocessor,
            env_preprocessor, env_postprocessor,
            suite_name, libero_task_id, args.n_eval_episodes, device,
            camera_name_mapping=_cam_map,
            videos_dir=videos_dir,
        )
        results[task_idx] = {"task_name": task_name, **metrics}
        log.info(f"  Success rate: {metrics['success_rate']:.1%}")

        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _log_eval_summary(results)
    return results


def eval_tail(
    args: argparse.Namespace,
    lora_lib: LoRALibrary,
    ds_meta: LeRobotDatasetMetadata,
    task_indices: list[int],
    device: torch.device,
    eval_output_dir: Path | None = None,
) -> dict:
    """Evaluate TAIL: for each task, load base + corresponding LoRA adapter."""
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME

    suite_name = args.eval_suite or args.dataset_repo_id.split("/")[-1].replace("_image", "")
    _cam_map = {"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}
    results = {}

    log.info("  Building LeRobot->LIBERO task_id mapping...")
    task_id_map = build_lerobot_to_libero_task_id_map(suite_name, ds_meta)

    log.info("  Pre-loading base model weights for eval...")
    pretrained_cache = cache_pretrained_weights(lora_lib.base_model)

    for task_idx in task_indices:
        adapter_info = lora_lib.get(task_idx)
        if adapter_info is None:
            log.warning(f"No LoRA adapter for task_idx={task_idx}. Skipping.")
            continue

        libero_task_id = task_id_map[task_idx]
        task_name = adapter_info["task_name"]
        adapter_path = Path(adapter_info["adapter_path"])
        log.info(f"\n[Eval-TAIL] task_idx={task_idx} -> LIBERO task_id={libero_task_id} | {task_name!r}")
        log.info(f"  Base model: {lora_lib.base_model}")
        log.info(f"  Adapter: {adapter_path}")

        base_policy = load_base_policy(lora_lib.base_model, ds_meta, device, args=args, _cached=pretrained_cache)
        peft_policy = load_lora_adapter(base_policy, adapter_path)
        merged_policy = merge_lora_into_policy(peft_policy)
        merged_policy.eval()

        preprocessor = PolicyProcessorPipeline.from_pretrained(
            str(adapter_path), config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            str(adapter_path), config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

        env_cfg = LiberoEnvConfig(task=suite_name, task_ids=[libero_task_id], camera_name_mapping=_cam_map)
        from lerobot.policies.xvla.processor_xvla import LiberoProcessorStep
        from lerobot.processor import PolicyProcessorPipeline as _PPL
        env_preprocessor = _PPL(steps=[LiberoProcessorStep()])
        env_postprocessor = _PPL(steps=[])  # identity: auto mode already outputs 7D axis-angle

        _vdir_root = eval_output_dir if eval_output_dir is not None else Path(args.output_dir)
        videos_dir = (_vdir_root / "eval_videos" / f"task{task_idx}") if args.save_videos else None
        metrics = _run_rollout_episodes(
            merged_policy, preprocessor, postprocessor,
            env_preprocessor, env_postprocessor,
            suite_name, libero_task_id, args.n_eval_episodes, device,
            camera_name_mapping=_cam_map,
            videos_dir=videos_dir,
        )
        results[task_idx] = {"task_name": task_name, **metrics}
        log.info(f"  Success rate: {metrics['success_rate']:.1%}")

        del base_policy, peft_policy, merged_policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _log_eval_summary(results)
    return results


def _log_eval_summary(results: dict) -> None:
    if not results:
        return
    log.info("\n" + "-"*60)
    log.info("Evaluation Summary:")
    per_task_sr = []
    for task_idx, info in sorted(results.items(), key=lambda x: x[0]):
        sr = info.get("success_rate", 0.0)
        per_task_sr.append(sr)
        log.info(f"  Task {task_idx} ({info.get('task_name','')[:40]}): "
                 f"{sr:.1%}  ({info.get('n_success',0)}/{info.get('n_episodes',0)})")
    mean_sr = sum(per_task_sr) / len(per_task_sr) if per_task_sr else 0.0
    log.info(f"  {'─'*49}")
    log.info(f"  Mean success rate: {mean_sr:.1%}  (across {len(per_task_sr)} tasks)")
    log.info("-"*60)


def save_eval_results(results: dict, output_dir: Path, suffix: str = "") -> None:
    path = output_dir / f"eval_results{suffix}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Eval results saved -> {path}")


# ══════════════════════════════════════════════════════════════════
# Main orchestration
# ══════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> None:
    is_ddp, local_rank, rank, world_size = init_ddp()
    is_main = (rank == 0)

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    init_logging()
    if not is_main:
        logging.getLogger().setLevel(logging.WARNING)
    set_seed(args.seed)

    gpu_ids = parse_gpus(args.gpus)
    device = torch.device(f"cuda:{local_rank}") if is_ddp else primary_device(gpu_ids)

    if is_main:
        log.info(f"CL method : {args.cl_method}")
        log.info(f"LoRA      : {args.use_lora}")
        log.info(f"DDP       : {is_ddp}  (world_size={world_size})")
        log.info(f"GPUs      : {gpu_ids if gpu_ids else 'CPU'}  device: {device}")

    _wandb_log = None
    if is_main and args.wandb and not args.eval_rollout:
        try:
            import wandb
            run_name = args.wandb_run_name or f"{args.cl_method}_{Path(args.output_dir).name}"
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
                dir=str(output_dir),
            )
            _wandb_log = wandb.log
            log.info(f"W&B run: {wandb.run.url}")
        except ImportError:
            log.warning("wandb not installed — skipping W&B logging. Run: pip install wandb")

    cl_state_path = output_dir / "cl_state.json"

    if args.resume and cl_state_path.exists():
        with open(cl_state_path) as f:
            cl_state = json.load(f)
        if not args.dataset_repo_id:
            args.dataset_repo_id = cl_state.get("dataset_repo_id", args.dataset_repo_id)
        if not args.pretrained:
            args.pretrained = cl_state.get("pretrained_start", args.pretrained)
        log.info(f"Resuming. Tasks done: {len(cl_state['tasks_completed'])}")
    else:
        cl_state = {
            "cl_method": args.cl_method,
            "use_lora": args.use_lora,
            "pretrained_start": args.pretrained,
            "dataset_repo_id": args.dataset_repo_id,
            "steps_per_task": args.steps_per_task,
            "seed": args.seed,
            "tasks_completed": [],
            "task_checkpoints": {},
            "task_stats": {},
        }

    log.info(f"Loading dataset metadata: {args.dataset_repo_id}")
    ds_meta = LeRobotDatasetMetadata(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
    )
    log.info(f"Total tasks: {len(ds_meta.tasks)}, fps={ds_meta.fps}")

    if args.suite:
        task_indices = resolve_suite(args.suite)
        log.info(f"Suite: {args.suite!r} -> task_indices {task_indices}")
    elif args.task_indices:
        task_indices = list(args.task_indices)
    else:
        task_indices = list(range(len(ds_meta.tasks)))
    log.info(f"CL sequence ({len(task_indices)} tasks): {task_indices}")

    if args.cl_method == "tail" and not args.use_lora and not args.eval_rollout:
        raise ValueError("--cl_method tail requires --use_lora.")

    _ddp_kwargs = dict(device=device, rank=rank, world_size=world_size,
                       local_rank=local_rank, is_main=is_main)
    if not args.eval_rollout:
        if args.cl_method == "sequential":
            cl_state = run_sequential(args, ds_meta, task_indices, output_dir, cl_state,
                                      wandb_log=_wandb_log, **_ddp_kwargs)
        elif args.cl_method == "tail":
            cl_state, lora_lib = run_tail(args, ds_meta, task_indices, output_dir, cl_state,
                                          wandb_log=_wandb_log, **_ddp_kwargs)
        else:
            raise ValueError(f"Unknown --cl_method: {args.cl_method!r}. Choose 'sequential' or 'tail'.")

    if args.eval_rollout and is_main:
        log.info("\n" + "="*60)
        log.info("Starting rollout evaluation...")
        if not os.environ.get("MUJOCO_GL"):
            os.environ["MUJOCO_GL"] = "egl"
        try:
            import libero  # noqa: F401
        except ImportError:
            raise ImportError(
                "LIBERO must be installed for rollout evaluation.\n"
                "  pip install hf-libero  or  see https://github.com/huggingface/lerobot"
            )

        eval_device = device
        eval_output_dir = Path(args.eval_output_dir) if args.eval_output_dir else None
        if eval_output_dir is not None:
            eval_output_dir.mkdir(parents=True, exist_ok=True)
            log.info(f"  Eval results will be saved to: {eval_output_dir}")
        method = cl_state.get("cl_method", args.cl_method)
        if method == "tail":
            library_dir = output_dir / "lora_library"
            lora_lib = LoRALibrary(library_dir)
            results = eval_tail(args, lora_lib, ds_meta, task_indices, eval_device,
                                eval_output_dir=eval_output_dir)
        else:
            results = eval_sequential(args, cl_state, ds_meta, task_indices, output_dir, eval_device,
                                      eval_output_dir=eval_output_dir)

        save_eval_results(results, eval_output_dir or output_dir, suffix=f"_{method}")

    if is_main:
        log.info("\n" + "="*60)
        log.info("Done.")
        if not args.eval_rollout:
            log.info(f"Tasks completed: {len(cl_state['tasks_completed'])}")
        log.info(f"Output dir: {output_dir}")

    if is_main and _wandb_log is not None:
        import wandb
        wandb.finish()

    if is_ddp:
        dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Continual Learning training of X-VLA on LIBERO (sequential or TAIL)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ──────────────────────────────────────────────────────
    p.add_argument("--pretrained", default="lerobot/xvla-base",
                   help="Initial XVLA checkpoint (HF Hub ID or local path).")

    # ── Dataset ───────────────────────────────────────────────────
    p.add_argument("--dataset_repo_id", default="lerobot/libero_10",
                   help="LeRobot-format LIBERO dataset.")
    p.add_argument("--dataset_root", default=None,
                   help="Optional local dataset root (skips Hub download).")
    p.add_argument("--task_indices", type=int, nargs="+", default=None,
                   help="Specific task indices to train on (in order). Default: all tasks.")
    p.add_argument("--suite", default=None,
                   choices=["libero_spatial", "libero_object", "libero_goal", "libero_10",
                            "spatial", "object", "goal", "long", "libero_long"],
                   help="LIBERO suite to train on. Overrides --task_indices.")

    # ── CL method ────────────────────────────────────────────────
    p.add_argument("--cl_method", default="sequential", choices=["sequential", "tail"],
                   help="CL strategy: 'sequential' or 'tail'.")

    # ── LoRA ─────────────────────────────────────────────────────
    p.add_argument("--use_lora", action="store_true",
                   help="Enable LoRA fine-tuning. Required for --cl_method=tail.")
    p.add_argument("--lora_r", type=int, default=64,
                   help="LoRA rank. Official PEFT docs recommend r=64 for LIBERO.")
    p.add_argument("--lora_alpha", type=int, default=64,
                   help="LoRA alpha (scaling = alpha / r). Default 64 → scaling=1.0.")
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument(
        "--lora_target_modules",
        default="q_proj,v_proj",
        help="Comma-separated module name suffixes for LoRA injection (PEFT suffix matching). "
             "Default 'q_proj,v_proj' targets Florence2 BART encoder self-attention Q/V projections. "
             "Actual module path: policy.model.vlm.language_model.model.encoder.layers.N.self_attn.q_proj. "
             "Add 'k_proj,out_proj' for more capacity. "
             "Note: policy transformer uses 'attn.proj' (single proj), add it explicitly if needed.",
    )

    # ── Training ──────────────────────────────────────────────────
    p.add_argument("--output_dir", default="outputs/cl_xvla_libero")
    p.add_argument("--steps_per_task", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--decay_lr_ratio", type=float, default=0.025,
                   help="Cosine decay floor as fraction of peak LR.")
    p.add_argument("--grad_clip_norm", type=float, default=10.0)
    p.add_argument("--num_workers", type=int, default=4)

    # ── XVLA arch / training flags ────────────────────────────────
    p.add_argument("--chunk_size", type=int, default=30,
                   help="Number of predicted action steps. Pretrained xvla-base uses 30.")
    p.add_argument("--n_obs_steps", type=int, default=1,
                   help="Observation history length (XVLA default: 1).")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"],
                   help="Model dtype. bfloat16 saves memory; XVLA config default is float32.")
    p.add_argument("--freeze_vision_encoder", action="store_true",
                   help="Freeze Florence2 vision encoder (vision_tower). "
                        "Language encoder and policy transformer remain trainable.")
    p.add_argument("--freeze_language_encoder", action="store_true",
                   help="Freeze Florence2 language encoder. Analogous to --train_expert_only "
                        "in Pi05 — prevents VLM language representation drift across tasks. "
                        "Recommended for Sequential CL to reduce catastrophic forgetting.")
    p.add_argument("--phase2", action="store_true",
                   help="Phase II fine-tuning style: freeze both vision and language encoders, "
                        "train only the policy transformer + soft prompts (~311M / 35%% of params). "
                        "Shortcut for --freeze_vision_encoder --freeze_language_encoder. "
                        "Faster per-task adaptation with less VLM forgetting risk. "
                        "Equivalent to the XVLAConfig default comment 'freeze VLM, train policy'.")
    p.add_argument("--no_train_policy_transformer", action="store_true",
                   help="Freeze the SoftPromptedTransformer (policy head). "
                        "Only soft prompts would remain trainable. Rarely useful.")
    p.add_argument("--no_train_soft_prompts", action="store_true",
                   help="Freeze soft prompt embeddings. Not recommended for CL.")
    p.add_argument("--action_mode", default="auto",
                   choices=["ee6d", "joint", "auto"],
                   help="Action space type for LIBERO. 'auto' (recommended) infers real_dim from dataset "
                        "and trims model output accordingly. 'ee6d' is for 20D 6D-rotation format datasets "
                        "and is NOT compatible with lerobot/libero_10 (which has 7D axis-angle actions).")
    p.add_argument("--num_denoising_steps", type=int, default=10,
                   help="Flow-matching denoising steps at inference (XVLA default: 10).")
    p.add_argument("--tokenizer_max_length", type=int, default=128,
                   help="Max token length for task description. Pretrained default is 1024 with "
                        "pad_language_to=max_length, which makes vlm_features ~1073 tokens and "
                        "exceeds the policy transformer's max_len_seq=512. LIBERO task descriptions "
                        "are ~25-35 tokens; 128 is more than sufficient.")
    p.add_argument("--vlm_lr_scale", type=float, default=0.1,
                   help="LR multiplier for Florence2 (VLM) params in full fine-tuning. "
                        "Matches XVLAAdamWConfig default of 1/10 VLM LR. "
                        "In Phase II / freeze modes, VLM is frozen so this has no effect.")
    p.add_argument("--soft_prompt_lr_scale", type=float, default=1.0,
                   help="LR multiplier for soft_prompt_hub params. Default 1.0 (same as transformer). "
                        "Matches XVLAAdamWConfig.soft_prompt_lr_scale. "
                        "Set lower (e.g. 0.1) to warm up soft prompts gradually.")

    # ── Evaluation ────────────────────────────────────────────────
    p.add_argument("--eval_rollout", action="store_true",
                   help="Run online rollout evaluation instead of training.")
    p.add_argument("--n_eval_episodes", type=int, default=20,
                   help="Number of rollout episodes per task during evaluation.")
    p.add_argument("--save_videos", action="store_true",
                   help="Save rollout episode videos to {output_dir}/eval_videos/task<N>/.")
    p.add_argument("--eval_suite", default=None,
                   help="LIBERO suite name for evaluation (default: inferred from dataset_repo_id).")
    p.add_argument("--eval_task", type=int, default=None,
                   help="(Sequential only) Fix a single checkpoint (by task_order) to evaluate ALL tasks. "
                        "Measures catastrophic forgetting. Default (None) uses each task's own checkpoint.")
    p.add_argument("--eval_output_dir", default=None,
                   help="Directory to save eval results and videos. Defaults to --output_dir.")

    # ── Hardware ──────────────────────────────────────────────────
    p.add_argument("--gpus", default="0",
                   help="GPU ID for single-process training. "
                        "Use torchrun --nproc_per_node=N for multi-GPU DDP.")

    # ── Weights & Biases ─────────────────────────────────────────
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", default="cl-xvla-libero",
                   help="W&B project name.")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run name (default: {cl_method}_{output_dir_name}).")

    # ── Misc ─────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_freq", type=int, default=100)
    p.add_argument("--resume", action="store_true",
                   help="Skip completed tasks and continue from last checkpoint.")

    args = p.parse_args()
    # --phase2 is a shortcut for freezing the entire VLM backbone
    if args.phase2:
        args.freeze_vision_encoder = True
        args.freeze_language_encoder = True
    # Convert negation store_true flags to positive booleans for config
    args.train_policy_transformer = not args.no_train_policy_transformer
    args.train_soft_prompts = not args.no_train_soft_prompts
    return args


if __name__ == "__main__":
    run(parse_args())
