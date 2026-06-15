#!/usr/bin/env python3
"""
eval_tail_offline.py  —  Offline accuracy check for TAIL LoRA adapters.

No LIBERO simulator required. Loads each task's adapter, runs forward passes
on a held-out subset of the LeRobot dataset, and reports action-prediction
L2 loss per task. Confirms that the TAIL inference pipeline (base + adapter
load, merge, preprocessor) is wired correctly.

Usage:
    # Evaluate all tasks in the library:
    python examples/training/eval_tail_offline.py \
        --output_dir outputs/debug_ddp4_tail \
        --dataset_repo_id lerobot/libero_10 \
        --gpu 0 --n_batches 10 --batch_size 4

    # Evaluate only task 0, 1, 2 (e.g. after training 3 tasks):
    python examples/training/eval_tail_offline.py \
        --output_dir outputs/debug_ddp4_tail \
        --task_indices 0 1 2 \
        --gpu 0 --n_batches 10 --batch_size 4
"""

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ── Import helpers from the training script ───────────────────────────────────
_TRAIN_SCRIPT = Path(__file__).parent / "train_smolvla_cl_libero.py"
spec = importlib.util.spec_from_file_location("train_cl", _TRAIN_SCRIPT)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

build_full_dataset        = _mod.build_full_dataset
build_policy_features_from_dataset = _mod.build_policy_features_from_dataset
TaskFilterDataset         = _mod.TaskFilterDataset
load_base_policy          = _mod.load_base_policy
load_lora_adapter         = _mod.load_lora_adapter
merge_lora_into_policy    = _mod.merge_lora_into_policy
cache_pretrained_weights  = _mod.cache_pretrained_weights
# ─────────────────────────────────────────────────────────────────────────────

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import (
    POLICY_PREPROCESSOR_DEFAULT_NAME,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    force=True,  # override any handler set by imported libs (transformers, etc.)
)
log = logging.getLogger(__name__)


def collate_fn(batch):
    keys = batch[0].keys()
    result = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            result[k] = torch.stack(vals)
        elif isinstance(vals[0], str):
            result[k] = vals
        else:
            try:
                result[k] = torch.tensor(vals)
            except Exception:
                result[k] = vals
    return result


@torch.no_grad()
def eval_adapter_offline(
    base_policy,
    adapter_path: Path,
    task_dataset,
    device: torch.device,
    n_batches: int,
    batch_size: int,
) -> dict:
    """Load adapter, merge, run N batches, return loss stats."""
    # Load per-task preprocessor
    preprocessor = PolicyProcessorPipeline.from_pretrained(
        str(adapter_path),
        config_filename=f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json",
    )

    # Load LoRA adapter and merge
    peft_policy  = load_lora_adapter(base_policy, adapter_path)
    merged       = merge_lora_into_policy(peft_policy)
    merged.eval()
    merged.to(device)

    camera_keys = [k for k in task_dataset.meta.camera_keys]

    dataloader = DataLoader(
        task_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn,
    )

    losses = []
    t0 = time.perf_counter()
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break

        # Normalise images uint8 → float
        for k in camera_keys:
            if k in batch and isinstance(batch[k], torch.Tensor) and batch[k].dtype == torch.uint8:
                batch[k] = batch[k].float() / 255.0

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch = preprocessor(batch)

        out  = merged(batch)
        loss = out[0] if isinstance(out, tuple) else out
        if loss.dim() > 0:
            loss = loss.mean()
        losses.append(loss.item())

    elapsed = time.perf_counter() - t0
    del peft_policy, merged
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not losses:
        return {"mean_loss": float("nan"), "std_loss": float("nan"), "n_batches": 0, "elapsed_s": elapsed}

    mean = sum(losses) / len(losses)
    std  = (sum((x - mean) ** 2 for x in losses) / max(1, len(losses) - 1)) ** 0.5
    return {"mean_loss": mean, "std_loss": std, "n_batches": len(losses), "elapsed_s": elapsed}


def run(args):
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "lora_library" / "manifest.json"
    if not manifest_path.exists():
        log.error(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    base_model      = manifest["base_model"]
    dataset_repo_id = manifest.get("dataset_repo_id", args.dataset_repo_id)
    adapters        = manifest["adapters"]  # dict str(task_order) → info

    # Filter by task_indices if specified
    requested = set(args.task_indices) if args.task_indices else None
    if requested is not None:
        adapters = {k: v for k, v in adapters.items() if v["task_index"] in requested}
        missing = requested - {v["task_index"] for v in adapters.values()}
        if missing:
            log.warning(f"task_indices {sorted(missing)} not found in manifest — skipping.")

    device = torch.device(f"cuda:{args.gpu}") if torch.cuda.is_available() else torch.device("cpu")
    log.info(f"Device: {device}")
    log.info(f"Base model: {base_model}")
    log.info(f"Dataset: {dataset_repo_id}")
    if requested is not None:
        log.info(f"Evaluating task_indices: {sorted(requested)} ({len(adapters)} found)")
    else:
        log.info(f"Evaluating all {len(adapters)} tasks in manifest")

    # Load dataset metadata
    ds_meta = LeRobotDatasetMetadata(repo_id=dataset_repo_id, root=args.dataset_root)

    # Pre-load pretrained weights once (reused for all tasks)
    log.info("\nPre-loading base weights (shared across all tasks)…")
    pretrained_cache = cache_pretrained_weights(base_model)

    # Load full dataset once (all episodes)
    log.info("Loading full dataset…")
    # Derive chunk_size/n_obs_steps from the first adapter's config
    first_adapter_dir = Path(list(adapters.values())[0]["adapter_path"])
    chunk_size = 50   # SmolVLA default
    n_obs_steps = 1

    full_dataset = build_full_dataset(
        dataset_repo_id, ds_meta, chunk_size, n_obs_steps, args.dataset_root
    )

    results = {}
    log.info("\n" + "="*60)
    log.info("Starting offline evaluation (TAIL — per-task adapter)…")
    log.info("="*60)

    for _, info in sorted(adapters.items(), key=lambda x: int(x[0])):
        task_idx    = info["task_index"]
        task_name   = info["task_name"]
        adapter_path = Path(info["adapter_path"])

        log.info(f"\n[Task {task_idx}] {task_name!r}")
        log.info(f"  Adapter: {adapter_path}")

        if not adapter_path.exists():
            log.warning(f"  Adapter directory missing — skipping.")
            results[task_idx] = {"task_name": task_name, "mean_loss": float("nan"), "skipped": True}
            continue

        # Filter dataset to this task's frames
        task_dataset = TaskFilterDataset(full_dataset, task_idx)
        if task_dataset.num_frames == 0:
            log.warning(f"  No dataset frames for task {task_idx} — skipping.")
            results[task_idx] = {"task_name": task_name, "mean_loss": float("nan"), "skipped": True}
            continue

        # Load base policy with dataset-compatible features (uses cached weights)
        base_policy = load_base_policy(base_model, ds_meta, device, _cached=pretrained_cache)

        metrics = eval_adapter_offline(
            base_policy, adapter_path, task_dataset, device,
            n_batches=args.n_batches, batch_size=args.batch_size,
        )
        results[task_idx] = {"task_name": task_name, **metrics}

        log.info(
            f"  loss = {metrics['mean_loss']:.4f} ± {metrics['std_loss']:.4f}  "
            f"({metrics['n_batches']} batches, {metrics['elapsed_s']:.1f}s)"
        )

        del base_policy
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary
    log.info("\n" + "="*60)
    log.info("OFFLINE EVAL SUMMARY (TAIL)")
    log.info("="*60)
    log.info(f"{'Task':>4}  {'Loss':>8}  {'±Std':>8}  {'Batches':>7}  Task name")
    log.info("-"*75)
    valid_losses = []
    for task_idx, info in sorted(results.items()):
        loss = info.get("mean_loss", float("nan"))
        std  = info.get("std_loss", float("nan"))
        nb   = info.get("n_batches", 0)
        name = info["task_name"][:50]
        skipped = info.get("skipped", False)
        flag = " [SKIP]" if skipped else ""
        log.info(f"  {task_idx:>2}  {loss:>8.4f}  {std:>8.4f}  {nb:>7}{flag}  {name}")
        if not skipped and not (loss != loss):  # not NaN
            valid_losses.append(loss)

    if valid_losses:
        avg = sum(valid_losses) / len(valid_losses)
        log.info("-"*75)
        log.info(f"  {'avg':>2}  {avg:>8.4f}   (across {len(valid_losses)} tasks)")
    log.info("="*60)

    # Save results
    out_json = output_dir / "eval_offline_results.json"
    out_json.write_text(json.dumps(
        {str(k): v for k, v in results.items()}, indent=2, default=str
    ))
    log.info(f"\nResults saved → {out_json}")


def parse_args():
    p = argparse.ArgumentParser(description="Offline TAIL adapter evaluation")
    p.add_argument("--output_dir", required=True,
                   help="Directory containing lora_library/manifest.json")
    p.add_argument("--dataset_repo_id", default="lerobot/libero_10",
                   help="LeRobot dataset used during training")
    p.add_argument("--dataset_root", default=None,
                   help="Local dataset root (passed to LeRobotDataset)")
    p.add_argument("--pretrained", default="lerobot/smolvla_base",
                   help="Pretrained base model (overridden by manifest)")
    p.add_argument("--task_indices", type=int, nargs="+", default=None,
                   help="Task indices to evaluate (e.g. 0 1 2). Omit to evaluate all tasks.")
    p.add_argument("--gpu", type=int, default=0, help="GPU id to use")
    p.add_argument("--n_batches", type=int, default=10,
                   help="Number of batches to evaluate per task")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Batch size for eval forward passes")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
