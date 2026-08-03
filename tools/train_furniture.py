from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from scene_repair_v3 import (
    ActionRange,
    FurnitureLoss,
    FurnitureLossConfig,
    FurnitureRepairNetwork,
    FurnitureVocabularies,
    FrozenFunctionalPartnerRules,
    ModelConfig,
)
from scene_repair_v3.checkpoint import load_checkpoint, save_checkpoint
from scene_repair_v3.data import CorruptionConfig
from scene_repair_v3.dataset import FurnitureCorruptionDataset, make_collate_fn


def _distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def _broadcast_seed(value: str, rank: int, device: torch.device) -> int:
    seed = secrets.randbits(63) if value == "auto" and rank == 0 else (0 if value == "auto" else int(value))
    tensor = torch.tensor([seed], device=device, dtype=torch.long)
    if dist.is_initialized():
        dist.broadcast(tensor, src=0)
    return int(tensor.item())


def _reduce(values: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def _run_epoch(model, loader, loss_fn, device, optimizer=None, use_bf16=True, max_batches=0):
    training = optimizer is not None
    model.train(training)
    totals = torch.zeros(28, device=device, dtype=torch.float64)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, (graph, targets, _metadata) in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            graph = graph.to(device)
            targets = targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                output = model(graph)
                losses = loss_fn(output, targets, graph)
            if training:
                losses.total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            eligible = targets.eligibility
            predictions = output.action_type_logits.argmax(dim=-1)
            correct = ((predictions == targets.action_types) & eligible).sum()
            count = eligible.sum()
            translation_mask = eligible & ((targets.action_types == 1) | (targets.action_types == 3))
            rotation_mask = eligible & ((targets.action_types == 2) | (targets.action_types == 3))
            normalized = output.primary_delta_normalized
            predicted_delta = torch.stack((
                normalized[:, 0] * loss_fn.action_range.max_translation_m,
                normalized[:, 1] * loss_fn.action_range.max_translation_m,
                normalized[:, 2] * loss_fn.action_range.max_yaw_rad,
            ), dim=-1)
            translation_error = torch.linalg.vector_norm(
                predicted_delta[:, :2] - targets.delta_m_rad[:, :2], dim=-1
            )
            yaw_error = torch.atan2(
                torch.sin(predicted_delta[:, 2] - targets.delta_m_rad[:, 2]),
                torch.cos(predicted_delta[:, 2] - targets.delta_m_rad[:, 2]),
            ).abs()
            graph_ids = output.furniture_graph_index
            per_scene_correct = torch.ones(graph.num_graphs, dtype=torch.int32, device=device)
            furniture_correct = ((predictions == targets.action_types) | ~eligible).to(torch.int32)
            per_scene_correct.scatter_reduce_(
                0, graph_ids, furniture_correct,
                reduce="amin", include_self=True,
            )
            confusion = torch.bincount(
                targets.action_types[eligible] * 4 + predictions[eligible], minlength=16
            ).reshape(-1)
            totals[:12] += torch.tensor([
                float(losses.total.detach()) * max(int(count), 1),
                float(correct), float(count),
                float(translation_error[translation_mask].sum()), float(translation_mask.sum()),
                float(yaw_error[rotation_mask].sum()), float(rotation_mask.sum()),
                1.0, float(graph.num_graphs), float(per_scene_correct.sum()),
                float(per_scene_correct.numel()),
                float(losses.geometry.detach()) * max(int(count), 1),
            ], device=device, dtype=torch.float64)
            totals[12:] += confusion.to(torch.float64)
    totals = _reduce(totals)
    confusion = totals[12:].reshape(4, 4)
    true_positive = confusion.diag()
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / confusion.sum(dim=1).clamp_min(1)
    macro_f1 = (2 * precision * recall / (precision + recall).clamp_min(1e-12)).mean()
    return {
        "loss": float(totals[0] / totals[2].clamp_min(1)),
        "action_accuracy": float(totals[1] / totals[2].clamp_min(1)),
        "translation_mae_m": float(totals[3] / totals[4].clamp_min(1)),
        "yaw_mae_rad": float(totals[5] / totals[6].clamp_min(1)),
        "action_macro_f1": float(macro_f1),
        "whole_scene_action_success": float(totals[9] / totals[10].clamp_min(1)),
        "geometry_loss": float(totals[11] / totals[2].clamp_min(1)),
        "batches": int(totals[7]),
        "graphs": int(totals[8]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Furniture v1 with online full-scene corruptions")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--functional-rules", type=Path, required=True)
    parser.add_argument("--compatibility-rules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64, help="Per-rank scene batch size")
    parser.add_argument("--train-samples-per-scene", type=int, default=16)
    parser.add_argument("--val-samples-per-scene", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--geometry-weight", type=float, default=0.5)
    parser.add_argument("--dense-pose-weight", type=float, default=0.0)
    parser.add_argument("--relation-pose-weight", type=float, default=0.0)
    parser.add_argument("--geometry-clearance-m", type=float, default=0.12)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--base-seed", default="auto")
    parser.add_argument(
        "--label-mode",
        choices=(
            "observable_joint_v4",
            "observable_v3",
            "semantic_restore_v1",
            "semantic_v2",
            "legacy_random",
        ),
        default="observable_joint_v4",
    )
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--max-batches", type=int, default=0, help="Limit each epoch for smoke tests")
    args = parser.parse_args()

    rank, world_size, local_rank = _distributed()
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    seed = _broadcast_seed(args.base_seed, rank, device)
    torch.manual_seed(seed + rank)
    vocab = FurnitureVocabularies.load(args.vocab)
    rules = FrozenFunctionalPartnerRules.load(args.functional_rules, args.compatibility_rules)
    corruption = CorruptionConfig(label_mode=args.label_mode)
    train_data = FurnitureCorruptionDataset(
        args.manifest, split="train", vocabularies=vocab, rules=rules,
        base_seed=seed, samples_per_scene=args.train_samples_per_scene,
        corruption=corruption,
    )
    val_data = FurnitureCorruptionDataset(
        args.manifest, split="val", vocabularies=vocab, rules=rules,
        base_seed=seed + 17, samples_per_scene=args.val_samples_per_scene,
        corruption=corruption,
    )
    train_sampler = DistributedSampler(train_data, shuffle=True, seed=seed) if world_size > 1 else None
    val_sampler = DistributedSampler(val_data, shuffle=False) if world_size > 1 else None
    loader_options = dict(
        batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=device.type == "cuda", persistent_workers=False,
        collate_fn=make_collate_fn(),
    )
    train_loader = DataLoader(train_data, sampler=train_sampler, shuffle=train_sampler is None, **loader_options)
    val_loader = DataLoader(val_data, sampler=val_sampler, shuffle=False, **loader_options)

    action_range = ActionRange(max_translation_m=3.0, max_yaw_rad=math.pi)
    network = FurnitureRepairNetwork(vocab, ModelConfig()).to(device)
    if args.init_checkpoint is not None:
        load_checkpoint(args.init_checkpoint, network, action_range, map_location=device)
    model = DistributedDataParallel(network, device_ids=[local_rank]) if world_size > 1 else network
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    loss_fn = FurnitureLoss(
        action_range,
        FurnitureLossConfig(
            translation_weight=2.0,
            keep_zero_weight=0.05,
            dense_pose_weight=args.dense_pose_weight,
            relation_pose_weight=args.relation_pose_weight,
            geometry_weight=args.geometry_weight,
            geometry_clearance_m=args.geometry_clearance_m,
        ),
    ).to(device)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run_config = vars(args).copy()
        deployment_action_policy = (
            "dense_delta" if args.label_mode == "semantic_restore_v1"
            else "hard_constraint_gated"
        )
        run_config.update({"base_seed_resolved": seed, "world_size": world_size,
                           "deployment_action_policy": deployment_action_policy,
                           "model_config": network.checkpoint_contract(),
                           "action_range": action_range.to_dict()})
        for key, value in list(run_config.items()):
            if isinstance(value, Path):
                run_config[key] = str(value)
        (args.output_dir / "run_config.json").write_text(
            json.dumps(run_config, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
    best_score = (-math.inf, -math.inf, -math.inf)
    log_path = args.output_dir / "metrics.jsonl"
    for epoch in range(args.epochs):
        train_data.set_epoch(epoch)
        val_data.set_epoch(0)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        start = time.time()
        train_metrics = _run_epoch(model, train_loader, loss_fn, device, optimizer,
                                   not args.no_bf16 and device.type == "cuda", args.max_batches)
        val_metrics = _run_epoch(model, val_loader, loss_fn, device, None,
                                 not args.no_bf16 and device.type == "cuda", args.max_batches)
        scheduler.step()
        record = {"epoch": epoch + 1, "seconds": time.time() - start,
                  "learning_rate": scheduler.get_last_lr()[0],
                  "train": train_metrics, "val": val_metrics}
        if rank == 0:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            print(json.dumps(record, ensure_ascii=True), flush=True)
            checkpoint_metadata = {
                "epoch": epoch + 1,
                "base_seed": seed,
                "label_mode": args.label_mode,
                "deployment_action_policy": deployment_action_policy,
                "dense_pose_weight": args.dense_pose_weight,
                "relation_pose_weight": args.relation_pose_weight,
            }
            save_checkpoint(args.output_dir / "last.pt", network, action_range,
                            optimizer=optimizer, extra=checkpoint_metadata)
            if args.label_mode == "semantic_restore_v1":
                score = (
                    val_metrics["whole_scene_action_success"],
                    -val_metrics["translation_mae_m"],
                    -val_metrics["loss"],
                )
            else:
                score = (
                    -val_metrics["geometry_loss"],
                    val_metrics["whole_scene_action_success"],
                    -val_metrics["translation_mae_m"],
                )
            if score > best_score:
                best_score = score
                save_checkpoint(args.output_dir / "best.pt", network, action_range,
                                optimizer=optimizer, extra={**checkpoint_metadata,
                                    "val_loss": val_metrics["loss"],
                                    "val_whole_scene_action_success": val_metrics["whole_scene_action_success"],
                                    "val_translation_mae_m": val_metrics["translation_mae_m"]})
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
