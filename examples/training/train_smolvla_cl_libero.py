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

"""Continual Learning (CL) training of SmolVLA on LIBERO benchmark.

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
    manifest.json                         ← task registry
    task_tidx0/adapter_model.safetensors  ← adapter weights only (~MBs)
    task_tidx0/policy_preprocessor.json   ← per-task normalization stats
    task_tidx1/...

Usage examples:

  # TAIL with LoRA (recommended for CL research):
  python train_smolvla_cl_libero.py \\
      --pretrained lerobot/smolvla_base \\
      --dataset_repo_id lerobot/libero_spatial \\
      --output_dir outputs/tail_smolvla_libero_spatial \\
      --cl_method tail --use_lora \\
      --steps_per_task 5000

  # Sequential with LoRA:
  python train_smolvla_cl_libero.py \\
      --dataset_repo_id lerobot/libero_10 \\
      --cl_method sequential --use_lora \\
      --lora_r 32 --steps_per_task 10000

  # Sequential without LoRA (full fine-tuning):
  python train_smolvla_cl_libero.py \\
      --dataset_repo_id lerobot/libero_10 \\
      --cl_method sequential --steps_per_task 5000

  # Evaluate TAIL library with rollouts (requires libero installed):
  python train_smolvla_cl_libero.py \\
      --output_dir outputs/tail_smolvla_libero_spatial \\
      --eval_rollout --n_eval_episodes 20 --resume

  # Resume interrupted training:
  python train_smolvla_cl_libero.py \\
      --output_dir outputs/tail_smolvla_libero_spatial \\
      --resume

Multi-GPU examples (DDP via torchrun — DataParallel is NOT supported):

  # 4-GPU DDP (TAIL):
  torchrun --nproc_per_node=4 train_smolvla_cl_libero.py --cl_method tail --use_lora ...

  # 2-GPU DDP (sequential):
  torchrun --nproc_per_node=2 train_smolvla_cl_libero.py --cl_method sequential ...

  # Single GPU:
  python train_smolvla_cl_libero.py --gpus 3 --cl_method tail --use_lora ...
"""

import argparse
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
from lerobot.policies.smolvla import SmolVLAConfig, SmolVLAPolicy
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


class _SmolVLALoRABridge(torch.nn.Module):
    """Bridge between PeftModel and SmolVLAPolicy.

    PeftModelForFeatureExtraction.forward has signature:
        forward(self, input_ids=None, attention_mask=None, ...)
    so when we call peft_model(batch_dict), Python binds batch_dict to
    `input_ids`. The PeftModel then calls base_model(input_ids=batch_dict, ...),
    which would fail on SmolVLAPolicy.forward(batch) because it doesn't accept
    keyword arguments.

    This wrapper's forward accepts input_ids and interprets it as the full
    batch dict, correctly routing it to SmolVLAPolicy.forward(batch).
    """

    def __init__(self, policy: SmolVLAPolicy):
        super().__init__()
        self.policy = policy
        # Expose config so get_base_config() can find SmolVLAConfig through PEFT layers
        self.config = policy.config

    def forward(self, input_ids=None, **kwargs):
        # PeftModelForFeatureExtraction calls: base_model(input_ids=batch_dict, ...)
        # where input_ids is actually our batch dict passed as a positional arg.
        if isinstance(input_ids, dict):
            return self.policy(input_ids)
        raise TypeError(
            f"_SmolVLALoRABridge.forward: expected batch dict in 'input_ids', "
            f"got {type(input_ids)}. Did you call peft_model(batch_dict)?"
        )


def apply_lora(
    policy: SmolVLAPolicy,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
) -> Any:
    """Wrap policy in a PEFT LoRA model; return PeftModel for training.

    Architecture:
        PeftModel
          └── LoraModel (BaseTuner)
                └── _SmolVLALoRABridge   ← PEFT-compatible forward bridge
                      └── SmolVLAPolicy  ← LoRA injected into target modules here

    The bridge is required because PeftModelForFeatureExtraction.forward binds
    the first positional argument to `input_ids`, but SmolVLAPolicy.forward
    expects a single batch dict.
    """
    _require_peft()
    from peft import LoraConfig, TaskType, get_peft_model

    bridge = _SmolVLALoRABridge(policy)
    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(bridge, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model


def save_lora_adapter(peft_model: Any, save_dir: Path) -> None:
    """Save only the LoRA adapter weights (tiny checkpoint)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(save_dir))
    log.info(f"LoRA adapter saved → {save_dir}")


def load_lora_adapter(base_policy: SmolVLAPolicy, adapter_dir: Path) -> Any:
    """Reconstruct a PeftModel from a saved LoRA adapter for inference."""
    _require_peft()
    from peft import PeftModel

    bridge = _SmolVLALoRABridge(base_policy)
    return PeftModel.from_pretrained(bridge, str(adapter_dir), is_trainable=False)


def merge_lora_into_policy(peft_model: Any) -> SmolVLAPolicy:
    """Merge LoRA weights into the base SmolVLAPolicy and return it.

    After merging, the returned SmolVLAPolicy can be used for inference or
    saved with save_pretrained() to produce a checkpoint the next task can
    load with SmolVLAPolicy.from_pretrained().
    """
    merged_bridge: _SmolVLALoRABridge = peft_model.merge_and_unload()
    # merge_and_unload() returns the inner module (our bridge)
    return merged_bridge.policy


def get_base_config(policy: Any) -> SmolVLAConfig:
    """Return SmolVLAConfig through DataParallel and/or PEFT wrappers."""
    p = unwrap_dp(policy)
    if hasattr(p, "base_model"):        # PeftModel → LoraModel → bridge → policy
        return p.base_model.model.config  # _SmolVLALoRABridge.config = SmolVLAConfig
    if isinstance(p, _SmolVLALoRABridge):
        return p.config
    return p.config


def get_trainable_params(policy: Any) -> list:
    return [p for p in policy.parameters() if p.requires_grad]


# ══════════════════════════════════════════════════════════════════
# Multi-GPU utilities
# ══════════════════════════════════════════════════════════════════

def parse_gpus(gpus_str: str) -> list[int]:
    """'0,1,2,3' → [0, 1, 2, 3].  Empty string → [] (use CPU)."""
    return [int(g.strip()) for g in gpus_str.split(",") if g.strip()]


def primary_device(gpu_ids: list[int]) -> torch.device:
    """Return the primary (first) CUDA device, or CPU if none."""
    if gpu_ids and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_ids[0]}")
    return torch.device("cpu")


def init_ddp() -> tuple[bool, int, int, int]:
    """Initialize torch.distributed if launched via torchrun.

    Returns (is_ddp, local_rank, rank, world_size).
    When not using torchrun (single process), returns (False, 0, 0, 1).
    Launch with: torchrun --nproc_per_node=4 train_smolvla_cl_libero.py ...
    """
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
    """Wrap policy with DistributedDataParallel when using torchrun.

    DataParallel is NOT used — SmolVLA's dict-based forward(batch) is incompatible
    with DP's **kwargs unpacking. DDP calls module(batch) directly and works fine.
    Use `torchrun --nproc_per_node=N` to enable multi-GPU training.
    """
    if is_ddp:
        # find_unused_parameters=True needed because some params may not contribute
        # to loss depending on the batch (e.g. empty_camera projections).
        return DDP(policy, device_ids=[local_rank], find_unused_parameters=True)
    if len(gpu_ids) > 1:
        log.warning(
            f"  --gpus {gpu_ids}: DataParallel disabled for SmolVLA. "
            f"Use `torchrun --nproc_per_node={len(gpu_ids)}` for multi-GPU DDP."
        )
    return policy


def unwrap_dp(policy: Any) -> Any:
    """Strip DataParallel or DistributedDataParallel wrapper."""
    if isinstance(policy, (torch.nn.DataParallel, DDP)):
        return policy.module
    return policy


def log_gpu_stats(gpu_ids: list[int], prefix: str = "") -> None:
    """Log current memory usage for each training GPU."""
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
    """Manages per-task LoRA adapters for the TAIL method.

    Manifest structure (manifest.json):
    {
      "base_model": "<hub_id_or_path>",
      "dataset_repo_id": "...",
      "adapters": {
        "<task_index>": {
          "task_index": int,
          "task_name": str,            # natural language task description
          "adapter_path": str,         # path to adapter weights dir
          "preprocessor_path": str,    # path to normalization stats
        }, ...
      }
    }
    """

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
# Suite → task index mapping  (HuggingFaceVLA/libero has 40 tasks)
# ══════════════════════════════════════════════════════════════════

SUITE_TASK_RANGES: dict[str, range] = {
    "libero_spatial": range(0, 10),   # spatial relations (same objects, diff positions)
    "libero_object":  range(10, 20),  # object manipulation (same scene, diff objects)
    "libero_goal":    range(20, 30),  # goal-conditioned  (same action, diff objects)
    "libero_10":      range(30, 40),  # long-horizon      (same object, diff locations)
}

SUITE_ALIASES: dict[str, str] = {
    "spatial":  "libero_spatial",
    "object":   "libero_object",
    "goal":     "libero_goal",
    "libero_goal": "libero_goal",
    "libero_spatial": "libero_spatial",
    "libero_object":  "libero_object",
    "long":     "libero_10",
    "libero_long": "libero_10",
    "libero_10": "libero_10",
}


def resolve_suite(suite_name: str) -> list[int]:
    """Return task indices for a named LIBERO suite."""
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
    """Derive PolicyFeature dicts (input, output) from a LeRobotDatasetMetadata.

    Video/image features use CHW ordering: (C, H, W).
    Dataset metadata stores them in HWC ordering: [H, W, C], so we transpose.
    """
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
            # Dataset shape is HWC [H, W, C] → convert to CHW (C, H, W)
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
    """Load pretrained SmolVLA once and return (config, cpu_state_dict).

    Used by run_tail to avoid reloading the same pretrained checkpoint for
    every task (10 tasks × ~1GB load = significant waste).
    """
    log.info(f"  Pre-loading pretrained weights (will be reused for all tasks): {pretrained}")
    base = SmolVLAPolicy.from_pretrained(pretrained)
    config = base.config
    state = {k: v.cpu() for k, v in base.state_dict().items()}
    del base
    torch.cuda.empty_cache()
    return config, state


def load_base_policy(
    pretrained: str,
    ds_meta: LeRobotDatasetMetadata,
    device: torch.device,
    _cached: tuple | None = None,
) -> SmolVLAPolicy:
    """Load SmolVLAPolicy with features adapted to the dataset.

    lerobot/smolvla_base was trained with camera1/camera2/camera3 + state(6).
    lerobot/libero_10 has image/wrist_image + state(8) + action(7).
    This mismatch would cause a shape error in the action head layers.

    Pass `_cached=(config, state_dict)` from `cache_pretrained_weights()` to
    skip the expensive pretrained load when calling this in a tight loop (TAIL).
    """
    input_features, output_features = build_policy_features_from_dataset(ds_meta)
    log.info(f"  Dataset input_features : {list(input_features.keys())}")
    log.info(f"  Dataset output_features: {list(output_features.keys())}")

    if _cached is not None:
        config, base_state = _cached
        # config is shared — make a shallow copy before mutating features
        import copy
        config = copy.copy(config)
        # Skip VLM weight loading: state_dict already contains them
        config.load_vlm_weights = False
    else:
        log.info(f"  Loading pretrained SmolVLA: {pretrained}")
        base = SmolVLAPolicy.from_pretrained(pretrained)
        config = base.config
        base_state = {k: v.cpu() for k, v in base.state_dict().items()}
        del base
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Override config with dataset features
    config.input_features = input_features
    config.output_features = output_features

    # Re-instantiate model with dataset-appropriate config, load matching weights
    policy = SmolVLAPolicy(config)
    missing, unexpected = policy.load_state_dict(base_state, strict=False)
    if missing:
        log.info(f"  Randomly initialized (dim mismatch): {len(missing)} keys "
                 f"(e.g. {missing[:3]})")
    if unexpected:
        log.info(f"  Skipped (not in new model): {len(unexpected)} keys")

    return policy.to(device)


# ══════════════════════════════════════════════════════════════════
# Dataset helpers
# ══════════════════════════════════════════════════════════════════

def get_task_name(ds_meta: LeRobotDatasetMetadata, task_index: int) -> str:
    """Return natural-language description for a given task_index."""
    matches = ds_meta.tasks[ds_meta.tasks["task_index"] == task_index]
    return matches.index[0] if len(matches) > 0 else f"task_{task_index}"


def build_lerobot_to_libero_task_id_map(suite_name: str, ds_meta: LeRobotDatasetMetadata) -> dict[int, int]:
    """Map LeRobot task_index → LIBERO benchmark task_id by matching task language strings.

    LeRobot dataset and LIBERO benchmark use different task orderings for the same suite.
    This function resolves the mapping so rollout evaluation uses the correct simulator task.
    """
    from libero.libero import benchmark as libero_benchmark

    bm = libero_benchmark.get_benchmark_dict()
    if suite_name not in bm:
        raise ValueError(f"Suite '{suite_name}' not found in LIBERO benchmark. Available: {list(bm.keys())}")
    suite = bm[suite_name]()

    # LIBERO task_id → language (lower-stripped for matching)
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

    log.info("  LeRobot→LIBERO task_id mapping:")
    for lr_idx, lib_id in sorted(mapping.items()):
        log.info(f"    LeRobot task_index={lr_idx} → LIBERO task_id={lib_id}")
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
    """Download and cache the COMPLETE dataset (no episode filtering).

    lerobot/libero_10 stores all episodes in a single parquet + one video file
    per camera (≈600 MB total). Loading everything at once is efficient; task
    filtering is then done in-memory via TaskFilterDataset using the task_index
    column already present in each parquet row.
    """
    delta_ts = build_delta_timestamps(ds_meta, chunk_size, n_obs_steps)
    log.info(f"  Loading full dataset {repo_id!r} (one-time ~600 MB download)…")
    return LeRobotDataset(
        repo_id=repo_id,
        root=root,
        episodes=None,          # all episodes, no filter
        delta_timestamps=delta_ts,
    )


class TaskFilterDataset(torch.utils.data.Dataset):
    """Wraps a full LeRobotDataset and exposes only frames for one task_index.

    Filters using the `task_index` column stored directly in each parquet row,
    bypassing the broken episode-index → file-index mapping in HuggingFaceVLA/libero.
    """

    def __init__(self, full_dataset: LeRobotDataset, task_index: int):
        # Use the .hf_dataset property (ensures reader is loaded) to get raw PyArrow table.
        # task_index is stored as scalar int64 (shape=(1,) → datasets.Value), so
        # to_pylist() yields plain Python ints and direct comparison works.
        arrow = full_dataset.hf_dataset.data                 # raw PyArrow Table, no transform
        task_col = arrow.column("task_index").to_pylist()
        self._indices: list[int] = [i for i, t in enumerate(task_col) if t == task_index]
        self._full = full_dataset
        self.meta = full_dataset.meta                        # exposes camera_keys etc.
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
# Core training loop (shared by all CL methods)
# ══════════════════════════════════════════════════════════════════

def train_steps(
    policy: Any,                        # SmolVLAPolicy or PeftModel wrapping it
    dataset: torch.utils.data.Dataset,  # LeRobotDataset or TaskFilterDataset
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
    wandb_log: "Callable[[dict], None] | None" = None,  # optional wandb logging callback
    global_step_offset: int = 0,        # cumulative steps before this task (for wandb x-axis)
) -> dict[str, float]:
    """Train for `steps` gradient updates; return loss statistics.

    With DDP (world_size > 1), each rank processes its own data shard via
    DistributedSampler. Effective batch size = batch_size * world_size.
    Loss logging is performed on every rank; saves are guarded by the caller.
    """
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
    # For DDP, manually track epochs so DistributedSampler.set_epoch() is called
    # at each new pass over the data (ensures different shuffle order per epoch).
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
    params = get_trainable_params(policy)
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-10
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
    """Sequential CL: inherit previous task's weights each step."""
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

    # Download and cache the full dataset once (avoids broken per-episode download)
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

        # Load base policy with features adapted to the dataset
        policy = load_base_policy(current_path, ds_meta, device)

        # Optionally wrap with LoRA
        if args.use_lora:
            lora_targets = [m.strip() for m in args.lora_target_modules.split(",")]
            policy = apply_lora(policy, args.lora_r, args.lora_alpha, args.lora_dropout, lora_targets)

        # Wrap with DDP for multi-GPU training (after LoRA)
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

        # Sync all ranks before saving
        if world_size > 1:
            dist.barrier()

        # Only rank 0 saves checkpoints
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
            log.info(f"  Checkpoint → {ckpt_dir}")
        else:
            current_path = str(output_dir / f"task_{task_order:02d}_tidx{task_idx}")

        # Wait for rank 0 to finish writing before any rank loads next checkpoint
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
    """TAIL: per-task LoRA adapter, always trained on frozen base model.

    Each adapter is saved independently; no information leaks between tasks
    during training, so catastrophic forgetting is impossible by design.
    """
    gpu_ids = parse_gpus(args.gpus)
    if device is None:
        device = primary_device(gpu_ids)
    if is_main and gpu_ids:
        log.info(f"  GPUs: {gpu_ids}  local_rank: {local_rank}  world_size: {world_size}  device: {device}")
        log_gpu_stats([local_rank] if world_size > 1 else gpu_ids, "initial ")

    # Download and cache the full dataset once
    full_dataset = build_full_dataset(
        args.dataset_repo_id, ds_meta, args.chunk_size, args.n_obs_steps, args.dataset_root
    )

    library_dir = output_dir / "lora_library"
    lora_lib = LoRALibrary(library_dir)
    if is_main:
        lora_lib.set_meta(args.pretrained, args.dataset_repo_id)
    if world_size > 1:
        dist.barrier()

    # Pre-load pretrained weights once; reuse across all tasks to avoid
    # reloading ~1GB of VLM weights N times (once per task in TAIL).
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

        # ── TAIL key step: always load from frozen BASE model ──────
        base_policy = load_base_policy(args.pretrained, ds_meta, device, _cached=pretrained_cache)

        # Apply fresh LoRA on top of frozen base
        lora_targets = [m.strip() for m in args.lora_target_modules.split(",")]
        policy = apply_lora(base_policy, args.lora_r, args.lora_alpha, args.lora_dropout, lora_targets)

        # Wrap with DDP for multi-GPU training (after LoRA)
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

        # Sync all ranks before saving
        if world_size > 1:
            dist.barrier()

        # Only rank 0 saves adapter and registers in library
        if is_main:
            adapter_dir = library_dir / f"task_tidx{task_idx}"
            save_lora_adapter(unwrap_dp(policy), adapter_dir)
            preprocessor.save_pretrained(adapter_dir)
            postprocessor.save_pretrained(adapter_dir)
            lora_lib.register(task_idx, task_name, adapter_dir, adapter_dir)
            _update_cl_state(cl_state, task_order, task_idx, task_name, task_dataset,
                             str(adapter_dir), train_stats, output_dir)
            log.info(f"  LoRA adapter → {adapter_dir}")
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
    libero_task_id: int,          # LIBERO benchmark task_id (≠ LeRobot task_index!)
    n_episodes: int,
    device: torch.device,
    obs_height: int = 360,
    obs_width: int = 360,
    camera_name_mapping: dict | None = None,
    videos_dir: Path | None = None,
) -> dict[str, float]:
    """Run rollout episodes on a single LIBERO task; return success metrics.

    Args:
        libero_task_id: The LIBERO benchmark task_id, which differs from the LeRobot
            dataset task_index. Use build_lerobot_to_libero_task_id_map() to convert.
    """
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

        # Save episode video
        if videos_dir is not None and ep_frames:
            from lerobot.utils.io_utils import write_video
            videos_dir.mkdir(parents=True, exist_ok=True)
            outcome = "success" if ep_success else "fail"
            video_path = videos_dir / f"ep{ep + 1:02d}_{outcome}.mp4"
            write_video(str(video_path), ep_frames, render_fps)
            log.info(f"    episode {ep+1}/{n_episodes}: {'SUCCESS' if ep_success else 'fail'}  → {video_path.name}")
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
) -> dict:
    """Evaluate sequential CL: each task uses its own saved full checkpoint."""
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors

    results = {}
    suite_name = args.eval_suite or args.dataset_repo_id.split("/")[-1].replace("_image", "")

    # Build task_index → LIBERO task_id mapping (ordering differs between LeRobot and LIBERO)
    log.info("  Building LeRobot→LIBERO task_id mapping…")
    task_id_map = build_lerobot_to_libero_task_id_map(suite_name, ds_meta)

    for task_order, task_idx in enumerate(task_indices):
        ckpt_key = str(task_order)
        if ckpt_key not in cl_state["task_checkpoints"]:
            log.warning(f"No checkpoint for task_order={task_order} (task_idx={task_idx}). Skipping eval.")
            continue

        libero_task_id = task_id_map[task_idx]
        ckpt_path = cl_state["task_checkpoints"][ckpt_key]
        task_name = get_task_name(ds_meta, task_idx)
        log.info(f"\n[Eval-Sequential] task_idx={task_idx} → LIBERO task_id={libero_task_id} | {task_name!r}")
        log.info(f"  Loading: {ckpt_path}")

        policy = SmolVLAPolicy.from_pretrained(ckpt_path).to(device)
        policy.eval()

        # Preprocessors saved alongside the checkpoint
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
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy.config)

        videos_dir = (output_dir / "eval_videos" / f"task{task_idx}") if args.save_videos else None
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
) -> dict:
    """Evaluate TAIL: for each task, load base + corresponding LoRA adapter."""
    from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.utils.constants import POLICY_PREPROCESSOR_DEFAULT_NAME, POLICY_POSTPROCESSOR_DEFAULT_NAME

    suite_name = args.eval_suite or args.dataset_repo_id.split("/")[-1].replace("_image", "")
    # camera key mapping: simulator names → dataset/policy names used in training
    _cam_map = {"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}
    results = {}

    # Build task_index → LIBERO task_id mapping (ordering differs between LeRobot and LIBERO)
    log.info("  Building LeRobot→LIBERO task_id mapping…")
    task_id_map = build_lerobot_to_libero_task_id_map(suite_name, ds_meta)

    # Pre-load base weights once; reused for every task adapter
    log.info("  Pre-loading base model weights for eval…")
    pretrained_cache = cache_pretrained_weights(lora_lib.base_model)

    for task_idx in task_indices:
        adapter_info = lora_lib.get(task_idx)
        if adapter_info is None:
            log.warning(f"No LoRA adapter for task_idx={task_idx}. Skipping.")
            continue

        libero_task_id = task_id_map[task_idx]
        task_name = adapter_info["task_name"]
        adapter_path = Path(adapter_info["adapter_path"])
        log.info(f"\n[Eval-TAIL] task_idx={task_idx} → LIBERO task_id={libero_task_id} | {task_name!r}")
        log.info(f"  Base model: {lora_lib.base_model}")
        log.info(f"  Adapter: {adapter_path}")

        # Load frozen base model (uses cached weights — no repeated download)
        base_policy = load_base_policy(lora_lib.base_model, ds_meta, device, _cached=pretrained_cache)

        # Load task-specific LoRA adapter, merge into SmolVLAPolicy for fast inference
        peft_policy = load_lora_adapter(base_policy, adapter_path)
        merged_policy = merge_lora_into_policy(peft_policy)
        merged_policy.eval()

        # Load per-task preprocessor (task-specific normalization stats)
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            str(adapter_path), config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
        )
        # postprocessor unnormalizes actions; input is PolicyAction (Tensor), output must also be Tensor
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            str(adapter_path), config_filename=f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

        env_cfg = LiberoEnvConfig(task=suite_name, task_ids=[libero_task_id], camera_name_mapping=_cam_map)
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, merged_policy.config)

        videos_dir = (Path(args.output_dir) / "eval_videos" / f"task{task_idx}") if args.save_videos else None
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
    log.info("\n" + "─"*60)
    log.info("Evaluation Summary:")
    per_task_sr = []
    for task_idx, info in sorted(results.items(), key=lambda x: x[0]):
        sr = info.get("success_rate", 0.0)
        per_task_sr.append(sr)
        log.info(f"  Task {task_idx} ({info.get('task_name','')[:40]}): "
                 f"{sr:.1%}  ({info.get('n_success',0)}/{info.get('n_episodes',0)})")
    mean_sr = sum(per_task_sr) / len(per_task_sr) if per_task_sr else 0.0
    log.info(f"  ─────────────────────────────────────────────────")
    log.info(f"  Mean success rate: {mean_sr:.1%}  (across {len(per_task_sr)} tasks)")
    log.info("─"*60)


def save_eval_results(results: dict, output_dir: Path, suffix: str = "") -> None:
    path = output_dir / f"eval_results{suffix}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Eval results saved → {path}")


# ══════════════════════════════════════════════════════════════════
# Main orchestration
# ══════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> None:
    # ── DDP initialization ───────────────────────────────────────
    # Single-process: init_ddp() returns (False, 0, 0, 1), device from --gpus.
    # torchrun: each process gets its local_rank; device = cuda:<local_rank>.
    is_ddp, local_rank, rank, world_size = init_ddp()
    is_main = (rank == 0)

    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    init_logging()
    # 非主进程只打印 WARNING 以上的日志，避免4份日志混在一起
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

    # ── Weights & Biases ────────────────────────────────────────
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

    # ── Resume ───────────────────────────────────────────────────
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

    # ── Dataset metadata ─────────────────────────────────────────
    if not args.eval_rollout or not cl_state.get("tasks_completed"):
        # Only need metadata during training (or if no tasks are done yet)
        pass

    log.info(f"Loading dataset metadata: {args.dataset_repo_id}")
    ds_meta = LeRobotDatasetMetadata(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
    )
    log.info(f"Total tasks: {len(ds_meta.tasks)}, fps={ds_meta.fps}")

    if args.suite:
        task_indices = resolve_suite(args.suite)
        log.info(f"Suite: {args.suite!r} → task_indices {task_indices}")
    elif args.task_indices:
        task_indices = list(args.task_indices)
    else:
        task_indices = list(range(len(ds_meta.tasks)))
    log.info(f"CL sequence ({len(task_indices)} tasks): {task_indices}")

    # ── 参数校验 ─────────────────────────────────────────────────
    if args.cl_method == "tail" and not args.use_lora:
        raise ValueError("--cl_method tail 必须配合 --use_lora 使用。")

    # ── Training phase ────────────────────────────────────────────
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

    # ── Evaluation phase ──────────────────────────────────────────
    # eval 只在 rank 0 跑：环境交互、视频录制、结果写文件都是单进程操作
    if args.eval_rollout and is_main:
        log.info("\n" + "="*60)
        log.info("Starting rollout evaluation…")
        # Headless OpenGL rendering via EGL (required on GPU servers without a display)
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
        method = cl_state.get("cl_method", args.cl_method)
        if method == "tail":
            library_dir = output_dir / "lora_library"
            lora_lib = LoRALibrary(library_dir)
            results = eval_tail(args, lora_lib, ds_meta, task_indices, eval_device)
        else:  # sequential
            results = eval_sequential(args, cl_state, ds_meta, task_indices, output_dir, eval_device)

        save_eval_results(results, output_dir, suffix=f"_{method}")

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
        description="Continual Learning training of SmolVLA on LIBERO (sequential or TAIL)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ──────────────────────────────────────────────────────
    p.add_argument("--pretrained", default="lerobot/smolvla_base",
                   help="Initial SmolVLA checkpoint (HF Hub ID or local path).")

    # ── Dataset ───────────────────────────────────────────────────
    p.add_argument("--dataset_repo_id", default="lerobot/libero_10",
                   help="LeRobot-format LIBERO dataset. Default: lerobot/libero_10 (10 long-horizon "
                        "tasks, ~600 MB). For multi-suite experiments use HuggingFaceVLA/libero "
                        "(40 tasks, much larger) together with --suite.")
    p.add_argument("--dataset_root", default=None,
                   help="Optional local dataset root (skips Hub download).")
    p.add_argument("--task_indices", type=int, nargs="+", default=None,
                   help="Specific task indices to train on (in order). Default: all tasks in dataset.")
    p.add_argument("--suite", default=None,
                   choices=["libero_spatial", "libero_object", "libero_goal", "libero_10",
                            "spatial", "object", "goal", "long", "libero_long"],
                   help="LIBERO suite to train on. Only meaningful with HuggingFaceVLA/libero (40 tasks). "
                        "Maps to task indices: spatial→0-9, object→10-19, goal→20-29, libero_10→30-39. "
                        "Overrides --task_indices.")

    # ── CL method ────────────────────────────────────────────────
    p.add_argument("--cl_method", default="sequential", choices=["sequential", "tail"],
                   help="CL strategy: 'sequential' (inherit prev ckpt) or "
                        "'tail' (per-task LoRA adapters on frozen base).")

    # ── LoRA ─────────────────────────────────────────────────────
    p.add_argument("--use_lora", action="store_true",
                   help="Enable LoRA fine-tuning. Required when --cl_method=tail; "
                        "optional for sequential (LoRA is merged into the checkpoint after each task).")
    p.add_argument("--lora_r", type=int, default=16,
                   help="LoRA rank (higher = more capacity, more parameters).")
    p.add_argument("--lora_alpha", type=int, default=32,
                   help="LoRA alpha (scaling factor = alpha / r).")
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="q_proj,v_proj",
                   help="Comma-separated list of module names to apply LoRA to. "
                        "For SmolVLA: 'q_proj,v_proj' targets VLM attention. "
                        "Add 'k_proj,o_proj' for more capacity. "
                        "Use 'all-linear' to target all linear layers.")

    # ── Training ──────────────────────────────────────────────────
    p.add_argument("--output_dir", default="outputs/cl_smolvla_libero")
    p.add_argument("--steps_per_task", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--decay_lr_ratio", type=float, default=0.025,
                   help="Cosine decay floor as fraction of peak LR.")
    p.add_argument("--grad_clip_norm", type=float, default=10.0)
    p.add_argument("--num_workers", type=int, default=4)

    # ── SmolVLA arch (must match pretrained model) ────────────────
    p.add_argument("--chunk_size", type=int, default=50,
                   help="Number of predicted action steps (SmolVLA default: 50).")
    p.add_argument("--n_obs_steps", type=int, default=1,
                   help="Observation history length (SmolVLA default: 1).")

    # ── Evaluation ────────────────────────────────────────────────
    p.add_argument("--eval_rollout", action="store_true",
                   help="Run online rollout evaluation instead of training. "
                        "Requires LIBERO to be installed.")
    p.add_argument("--n_eval_episodes", type=int, default=20,
                   help="Number of rollout episodes per task during evaluation.")
    p.add_argument("--save_videos", action="store_true",
                   help="Save rollout episode videos to {output_dir}/eval_videos/task<N>/.")
    p.add_argument("--eval_suite", default=None,
                   help="LIBERO suite name for evaluation (default: inferred from dataset_repo_id). "
                        "E.g. 'libero_spatial', 'libero_10'.")

    # ── Hardware ──────────────────────────────────────────────────
    p.add_argument("--gpus", default="0",
                   help="GPU ID for single-process training (e.g. '0' or '3'). "
                        "For multi-GPU DDP use torchrun --nproc_per_node=N instead; "
                        "this flag is ignored in that case. CPU: '' (empty string).")

    # ── Weights & Biases ─────────────────────────────────────────
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging.")
    p.add_argument("--wandb_project", default="cl-smolvla-libero",
                   help="W&B project name.")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run name (default: {cl_method}_{output_dir_name}).")

    # ── Misc ─────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_freq", type=int, default=100)
    p.add_argument("--resume", action="store_true",
                   help="Skip completed tasks and continue from last checkpoint.")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
