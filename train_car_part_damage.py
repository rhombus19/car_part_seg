#!/usr/bin/env python3
"""
Train a UPerNet semantic segmentation model on the Hugging Face dataset:
    moondream/car_part_damage

Usage:
    huggingface-cli login
    python train_car_part_damage.py --epochs 20 --batch-size 4 --amp
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from PIL import Image, ImageDraw, ImageOps
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise SystemExit(
        "segmentation_models_pytorch is required for UPerNet. Install requirements.txt first."
    ) from exc


BACKGROUND_LABEL = "background"
PART_CLASSES: List[str] = [
    "Back-bumper",
    "Back-door",
    "Back-wheel",
    "Back-window",
    "Back-windshield",
    "Fender",
    "Front-bumper",
    "Front-door",
    "Front-wheel",
    "Front-window",
    "Grille",
    "Headlight",
    "Hood",
    "License-plate",
    "Mirror",
    "Quarter-panel",
    "Rocker-panel",
    "Roof",
    "Tail-light",
    "Trunk",
    "Windshield",
]
CLASS_NAMES: List[str] = [BACKGROUND_LABEL, *PART_CLASSES]
LABEL2ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID2LABEL: Dict[int, str] = {idx: name for name, idx in LABEL2ID.items()}
NUM_CLASSES = len(CLASS_NAMES)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train car-part semantic segmentation model.")
    parser.add_argument("--dataset-name", type=str, default="moondream/car_part_damage")
    parser.add_argument("--output-dir", type=str, default="artifacts/car_part_damage")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument(
        "--split-mode",
        type=str,
        default="resplit",
        choices=["resplit", "official"],
        help="Use a merged class-aware train/val resplit or keep the official train/test split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Validation ratio used when --split-mode=resplit.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used for the resplit.",
    )
    parser.add_argument("--encoder-name", type=str, default="tu-convnext_small")
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help="Use 'imagenet' to match pretrained timm-backed encoder weights, or 'none' for random init.",
    )
    parser.add_argument(
        "--best-metric",
        type=str,
        default="foreground_mean_iou",
        choices=["foreground_mean_iou", "mean_iou", "pixel_accuracy"],
        help="Validation metric used to keep the best checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument(
        "--train-samples",
        type=int,
        default=0,
        help="If >0, train on only the first N samples (useful for smoke tests).",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=0,
        help="If >0, evaluate on only the first N test samples.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def polygon_area(flat_coords: Sequence[float]) -> float:
    if len(flat_coords) < 6:
        return 0.0

    x = flat_coords[0::2]
    y = flat_coords[1::2]
    area = 0.0
    for idx in range(len(x)):
        next_idx = (idx + 1) % len(x)
        area += x[idx] * y[next_idx] - x[next_idx] * y[idx]
    return abs(area) * 0.5


def to_polygon_points(flat_coords: Sequence[float]) -> List[Tuple[float, float]]:
    return [(flat_coords[idx], flat_coords[idx + 1]) for idx in range(0, len(flat_coords), 2)]


def labels_from_annotations(annotations: Sequence[Dict[str, object]]) -> List[str]:
    return sorted({ann["category"] for ann in annotations if ann["category"] in LABEL2ID})


def combine_splits(parts: Sequence[Dataset]) -> Dataset:
    non_empty_parts = [part for part in parts if len(part) > 0]
    if not non_empty_parts:
        raise ValueError("Expected at least one non-empty dataset split")
    if len(non_empty_parts) == 1:
        return non_empty_parts[0]
    return concatenate_datasets(non_empty_parts)


def build_resplit(
    train_split: Dataset,
    test_split: Dataset,
    val_ratio: float,
    seed: int,
) -> Tuple[Dataset, Dataset, Dict[str, object]]:
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("val-ratio must be between 0 and 0.5 for resplit mode")

    source_splits = {"train": train_split, "test": test_split}
    class_image_counts: Counter[str] = Counter()
    records: List[Dict[str, object]] = []

    for source_name, split in source_splits.items():
        image_ids = split["image_id"] if "image_id" in split.column_names else None
        annotations_column = split["annotations"]
        for idx, annotations in enumerate(annotations_column):
            labels = labels_from_annotations(annotations)
            for label in labels:
                class_image_counts[label] += 1
            records.append(
                {
                    "source": source_name,
                    "index": idx,
                    "image_id": str(image_ids[idx]) if image_ids is not None else f"{source_name}_{idx:05d}",
                    "labels": labels,
                }
            )

    target_val_size = max(1, int(round(len(records) * val_ratio)))
    target_val_by_class = {
        label: min(count - 1, max(1, int(round(count * val_ratio)))) if count > 1 else 0
        for label, count in class_image_counts.items()
    }

    rng = random.Random(seed)
    rng.shuffle(records)
    records.sort(
        key=lambda record: (
            -sum(1.0 / class_image_counts[label] for label in record["labels"]),
            -len(record["labels"]),
        )
    )

    val_records: List[Dict[str, object]] = []
    val_record_keys = set()
    val_class_counts: Counter[str] = Counter()

    for record in records:
        if len(val_records) >= target_val_size:
            break
        if any(val_class_counts[label] < target_val_by_class[label] for label in record["labels"]):
            key = (record["source"], record["index"])
            if key in val_record_keys:
                continue
            val_record_keys.add(key)
            val_records.append(record)
            for label in record["labels"]:
                val_class_counts[label] += 1

    for record in records:
        if len(val_records) >= target_val_size:
            break
        key = (record["source"], record["index"])
        if key in val_record_keys:
            continue
        val_record_keys.add(key)
        val_records.append(record)
        for label in record["labels"]:
            val_class_counts[label] += 1

    val_indices_by_source = {"train": [], "test": []}
    for record in val_records:
        val_indices_by_source[record["source"]].append(int(record["index"]))
    for indices in val_indices_by_source.values():
        indices.sort()

    train_indices_by_source = {}
    for source_name, split in source_splits.items():
        val_index_set = set(val_indices_by_source[source_name])
        train_indices_by_source[source_name] = [
            idx for idx in range(len(split)) if idx not in val_index_set
        ]

    merged_train = combine_splits(
        [
            train_split.select(train_indices_by_source["train"]),
            test_split.select(train_indices_by_source["test"]),
        ]
    )
    merged_val = combine_splits(
        [
            train_split.select(val_indices_by_source["train"]),
            test_split.select(val_indices_by_source["test"]),
        ]
    )

    split_summary = {
        "strategy": "merged_multilabel_resplit",
        "seed": seed,
        "val_ratio": val_ratio,
        "train_size": len(merged_train),
        "val_size": len(merged_val),
        "source_sizes": {name: len(split) for name, split in source_splits.items()},
        "train_indices_by_source": train_indices_by_source,
        "val_indices_by_source": val_indices_by_source,
        "class_image_counts_total": dict(sorted(class_image_counts.items())),
        "class_image_counts_val": dict(sorted(val_class_counts.items())),
        "class_image_counts_train": {
            label: class_image_counts[label] - val_class_counts[label]
            for label in sorted(class_image_counts)
        },
        "val_records": val_records,
    }
    return merged_train, merged_val, split_summary


def build_segmentation_mask(example: Dict[str, object]) -> Image.Image:
    width = int(example["width"])
    height = int(example["height"])
    mask = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(mask)

    annotations = sorted(
        example["annotations"],
        key=lambda ann: polygon_area(ann["segmentation"][0]) if ann.get("segmentation") else 0.0,
        reverse=True,
    )

    for ann in annotations:
        class_name = ann["category"]
        if class_name not in LABEL2ID:
            raise ValueError(f"Unknown category in dataset: {class_name}")

        polygons = ann.get("segmentation") or []
        if not polygons:
            continue

        exterior = polygons[0]
        if len(exterior) < 6:
            continue

        class_id = LABEL2ID[class_name]
        draw.polygon(to_polygon_points(exterior), fill=class_id)

        for hole in polygons[1:]:
            if len(hole) >= 6:
                draw.polygon(to_polygon_points(hole), fill=0)

    return mask


def resize_and_pad(image: Image.Image, mask: Image.Image, target_size: int) -> Tuple[Image.Image, Image.Image]:
    if target_size <= 0:
        raise ValueError("image-size must be a positive integer")

    width, height = image.size
    scale = target_size / max(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    image = image.resize((new_width, new_height), resample=Image.BILINEAR)
    mask = mask.resize((new_width, new_height), resample=Image.NEAREST)

    pad_w = target_size - new_width
    pad_h = target_size - new_height
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)
    mask = ImageOps.expand(mask, border=(left, top, right, bottom), fill=0)
    return image, mask


class CarPartSegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, split: Dataset, image_size: int, train: bool, hflip_prob: float) -> None:
        self.split = split
        self.image_size = image_size
        self.train = train
        self.hflip_prob = hflip_prob

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        example = self.split[idx]
        image = example["image"].convert("RGB")
        mask = build_segmentation_mask(example)

        if self.train and self.hflip_prob > 0.0 and random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image, mask = resize_and_pad(image, mask, self.image_size)

        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, IMAGENET_MEAN, IMAGENET_STD)
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_t, mask_t


def collate_fn(batch: Iterable[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    images, masks = zip(*batch)
    return torch.stack(images), torch.stack(masks)


def build_model(encoder_name: str, encoder_weights: str | None) -> nn.Module:
    return smp.UPerNet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        classes=NUM_CLASSES,
        activation=None,
    )


def update_confusion_matrix(
    confusion: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> None:
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)
    valid = (targets >= 0) & (targets < num_classes)
    indices = num_classes * targets[valid] + preds[valid]
    confusion += torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def compute_metrics(confusion: torch.Tensor) -> Dict[str, object]:
    confusion = confusion.to(torch.float64)
    true_positives = confusion.diag()
    false_positives = confusion.sum(dim=0) - true_positives
    false_negatives = confusion.sum(dim=1) - true_positives
    union = true_positives + false_positives + false_negatives

    per_class_iou: Dict[str, float | None] = {}
    valid_scores: List[float] = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        union_value = union[class_idx].item()
        if union_value == 0:
            per_class_iou[class_name] = None
            continue
        iou = (true_positives[class_idx] / union[class_idx]).item()
        per_class_iou[class_name] = iou
        valid_scores.append(iou)

    pixel_accuracy = (true_positives.sum() / confusion.sum()).item() if confusion.sum() > 0 else 0.0
    mean_iou = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    foreground_scores = [score for name, score in per_class_iou.items() if name != BACKGROUND_LABEL and score is not None]
    foreground_mean_iou = sum(foreground_scores) / len(foreground_scores) if foreground_scores else 0.0

    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "foreground_mean_iou": foreground_mean_iou,
        "per_class_iou": per_class_iou,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    print_freq: int,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
) -> float:
    model.train()
    running_loss = 0.0
    num_batches = len(loader)

    for step, (images, masks) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

        if step % print_freq == 0 or step == num_batches:
            avg_so_far = running_loss / step
            print(f"epoch={epoch:02d} step={step:04d}/{num_batches:04d} train_loss={avg_so_far:.4f}")

    return running_loss / max(1, num_batches)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, masks)
        else:
            logits = model(images)
            loss = criterion(logits, masks)
        preds = logits.argmax(dim=1)

        total_loss += loss.item()
        update_confusion_matrix(confusion, preds, masks, NUM_CLASSES)

    metrics = compute_metrics(confusion.cpu())
    metrics["loss"] = total_loss / max(1, len(loader))
    return metrics


def save_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    output_dir: Path,
    args: argparse.Namespace,
    split_summary: Dict[str, object],
    metrics: Dict[str, object] | None = None,
    best: bool = False,
) -> None:
    filename = "checkpoint_best.pt" if best else f"checkpoint_epoch_{epoch:03d}.pt"
    ckpt_path = output_dir / filename
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "label2id": LABEL2ID,
            "id2label": ID2LABEL,
            "class_names": CLASS_NAMES,
            "model_config": {
                "arch": "UPerNet",
                "encoder_name": args.encoder_name,
                "encoder_weights": None if args.encoder_weights == "none" else args.encoder_weights,
                "classes": NUM_CLASSES,
                "image_size": args.image_size,
            },
            "metrics": metrics or {},
            "args": vars(args),
            "split_summary": split_summary,
        },
        ckpt_path,
    )
    print(f"saved checkpoint: {ckpt_path}")


def describe_split(name: str, split: Dataset) -> None:
    print(f"{name} images: {len(split)}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights

    print(f"loading dataset: {args.dataset_name}")
    ds: DatasetDict = load_dataset(args.dataset_name)
    official_train_split = ds["train"]
    official_test_split = ds["test"]

    if args.train_samples > 0:
        official_train_split = official_train_split.select(
            range(min(args.train_samples, len(official_train_split)))
        )
    if args.test_samples > 0:
        official_test_split = official_test_split.select(
            range(min(args.test_samples, len(official_test_split)))
        )

    if args.split_mode == "resplit":
        train_split, val_split, split_summary = build_resplit(
            official_train_split,
            official_test_split,
            val_ratio=args.val_ratio,
            seed=args.split_seed,
        )
    else:
        train_split = official_train_split
        val_split = official_test_split
        split_summary = {
            "strategy": "official_train_test",
            "train_size": len(train_split),
            "val_size": len(val_split),
            "source_sizes": {
                "train": len(official_train_split),
                "test": len(official_test_split),
            },
        }

    describe_split("train", train_split)
    describe_split("val", val_split)
    print(f"num classes: {NUM_CLASSES} ({NUM_CLASSES - 1} foreground + background)")
    print(f"split_mode={args.split_mode} best_metric={args.best_metric}")

    save_json(
        output_dir / "label_map.json",
        {
            "class_names": CLASS_NAMES,
            "label2id": LABEL2ID,
            "id2label": ID2LABEL,
        },
    )
    save_json(output_dir / "split_summary.json", split_summary)

    train_dataset = CarPartSegmentationDataset(
        train_split,
        image_size=args.image_size,
        train=True,
        hflip_prob=args.hflip_prob,
    )
    val_dataset = CarPartSegmentationDataset(
        val_split,
        image_size=args.image_size,
        train=False,
        hflip_prob=0.0,
    )

    common_loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_fn,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )

    model = build_model(args.encoder_name, encoder_weights=encoder_weights).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    criterion = nn.CrossEntropyLoss()

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    best_metric_value = float("-inf")
    history: List[Dict[str, object]] = []

    print(
        f"training on device={device} amp={use_amp} encoder={args.encoder_name} image_size={args.image_size}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            print_freq=args.print_freq,
            use_amp=use_amp,
            scaler=scaler,
        )

        epoch_metrics: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }

        if epoch % args.eval_every == 0:
            eval_metrics = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
            epoch_metrics.update(
                {
                    "val_loss": eval_metrics["loss"],
                    "pixel_accuracy": eval_metrics["pixel_accuracy"],
                    "mean_iou": eval_metrics["mean_iou"],
                    "foreground_mean_iou": eval_metrics["foreground_mean_iou"],
                }
            )
            print(
                "epoch={:02d} train_loss={:.4f} val_loss={:.4f} miou={:.4f} fg_miou={:.4f} pix_acc={:.4f}".format(
                    epoch,
                    train_loss,
                    eval_metrics["loss"],
                    eval_metrics["mean_iou"],
                    eval_metrics["foreground_mean_iou"],
                    eval_metrics["pixel_accuracy"],
                )
            )
            save_json(output_dir / f"val_metrics_epoch_{epoch:03d}.json", eval_metrics)

            selected_metric_value = float(eval_metrics[args.best_metric])
            epoch_metrics["best_metric_name"] = args.best_metric
            epoch_metrics["best_metric_value"] = selected_metric_value

            if selected_metric_value > best_metric_value:
                best_metric_value = selected_metric_value
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    output_dir,
                    args,
                    split_summary,
                    metrics=epoch_metrics,
                    best=True,
                )
        else:
            print(f"epoch={epoch:02d} train_loss={train_loss:.4f}")

        history.append(epoch_metrics)
        save_json(output_dir / "history.json", history)

        if epoch % args.checkpoint_every == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                output_dir,
                args,
                split_summary,
                metrics=epoch_metrics,
            )

        scheduler.step()

    final_path = output_dir / "model_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label2id": LABEL2ID,
            "id2label": ID2LABEL,
            "class_names": CLASS_NAMES,
            "model_config": {
                "arch": "UPerNet",
                "encoder_name": args.encoder_name,
                "encoder_weights": encoder_weights,
                "classes": NUM_CLASSES,
                "image_size": args.image_size,
            },
            "split_summary": split_summary,
        },
        final_path,
    )
    print(f"saved final model: {final_path}")


if __name__ == "__main__":
    main()
