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
import io
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision.transforms import ColorJitter, InterpolationMode
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
    parser.add_argument("--scale-min", type=float, default=0.75)
    parser.add_argument("--scale-max", type=float, default=1.50)
    parser.add_argument("--affine-prob", type=float, default=0.7)
    parser.add_argument("--crop-foreground-prob", type=float, default=0.8)
    parser.add_argument("--color-jitter-prob", type=float, default=0.8)
    parser.add_argument("--blur-prob", type=float, default=0.15)
    parser.add_argument("--jpeg-prob", type=float, default=0.10)
    parser.add_argument("--noise-prob", type=float, default=0.15)
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
        "--refit-on-full-data",
        action="store_true",
        help="After validation-based model selection, retrain from scratch on all available images.",
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
        help="If >0, use only the first N official test samples when building/evaluating splits.",
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


def build_splits_from_summary(ds: DatasetDict, split_summary: Dict[str, object]) -> Tuple[Dataset, Dataset]:
    official_train_split = ds["train"]
    official_test_split = ds["test"]
    strategy = split_summary["strategy"]

    if strategy == "merged_multilabel_resplit":
        train_indices_by_source = split_summary["train_indices_by_source"]
        val_indices_by_source = split_summary["val_indices_by_source"]
        train_split = combine_splits(
            [
                official_train_split.select(train_indices_by_source["train"]),
                official_test_split.select(train_indices_by_source["test"]),
            ]
        )
        val_split = combine_splits(
            [
                official_train_split.select(val_indices_by_source["train"]),
                official_test_split.select(val_indices_by_source["test"]),
            ]
        )
        return train_split, val_split

    if strategy == "official_train_test":
        train_indices_by_source = split_summary.get(
            "train_indices_by_source",
            {"train": list(range(len(official_train_split))), "test": []},
        )
        val_indices_by_source = split_summary.get(
            "val_indices_by_source",
            {"train": [], "test": list(range(len(official_test_split)))},
        )
        train_parts = []
        val_parts = []
        if train_indices_by_source["train"]:
            train_parts.append(official_train_split.select(train_indices_by_source["train"]))
        if train_indices_by_source["test"]:
            train_parts.append(official_test_split.select(train_indices_by_source["test"]))
        if val_indices_by_source["train"]:
            val_parts.append(official_train_split.select(val_indices_by_source["train"]))
        if val_indices_by_source["test"]:
            val_parts.append(official_test_split.select(val_indices_by_source["test"]))
        return combine_splits(train_parts), combine_splits(val_parts)

    raise ValueError(f"Unknown split strategy: {strategy}")


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


def resize_long_side(image: Image.Image, mask: Image.Image, target_long_side: int) -> Tuple[Image.Image, Image.Image]:
    width, height = image.size
    scale = target_long_side / max(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = image.resize((new_width, new_height), resample=Image.BILINEAR)
    mask = mask.resize((new_width, new_height), resample=Image.NEAREST)
    return image, mask


def pad_if_needed(image: Image.Image, mask: Image.Image, target_size: int) -> Tuple[Image.Image, Image.Image]:
    width, height = image.size
    pad_w = max(0, target_size - width)
    pad_h = max(0, target_size - height)
    if pad_w == 0 and pad_h == 0:
        return image, mask

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    image = ImageOps.expand(image, border=(left, top, right, bottom), fill=0)
    mask = ImageOps.expand(mask, border=(left, top, right, bottom), fill=0)
    return image, mask


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def random_crop_coordinates(mask_np: np.ndarray, crop_size: int, prefer_foreground_prob: float) -> Tuple[int, int]:
    height, width = mask_np.shape
    max_top = height - crop_size
    max_left = width - crop_size
    if max_top < 0 or max_left < 0:
        raise ValueError("Crop size must not exceed mask size after padding")

    foreground_positions = np.argwhere(mask_np > 0)
    use_foreground = len(foreground_positions) > 0 and random.random() < prefer_foreground_prob
    if use_foreground:
        center_y, center_x = foreground_positions[random.randrange(len(foreground_positions))]
        top_min = max(0, int(center_y) - crop_size + 1)
        top_max = min(int(center_y), max_top)
        left_min = max(0, int(center_x) - crop_size + 1)
        left_max = min(int(center_x), max_left)
        top = random.randint(top_min, top_max) if top_min < top_max else top_min
        left = random.randint(left_min, left_max) if left_min < left_max else left_min
        return top, left

    top = random.randint(0, max_top) if max_top > 0 else 0
    left = random.randint(0, max_left) if max_left > 0 else 0
    return top, left


class SegmentationTrainAugmenter:
    def __init__(
        self,
        image_size: int,
        hflip_prob: float,
        scale_min: float,
        scale_max: float,
        affine_prob: float,
        crop_foreground_prob: float,
        color_jitter_prob: float,
        blur_prob: float,
        jpeg_prob: float,
        noise_prob: float,
    ) -> None:
        self.image_size = image_size
        self.hflip_prob = hflip_prob
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.affine_prob = affine_prob
        self.crop_foreground_prob = crop_foreground_prob
        self.color_jitter_prob = color_jitter_prob
        self.blur_prob = blur_prob
        self.jpeg_prob = jpeg_prob
        self.noise_prob = noise_prob
        self.color_jitter = ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.02,
        )

    def __call__(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        if random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        target_long_side = int(round(self.image_size * random.uniform(self.scale_min, self.scale_max)))
        image, mask = resize_long_side(image, mask, max(1, target_long_side))

        if random.random() < self.affine_prob:
            width, height = image.size
            angle = random.uniform(-8.0, 8.0)
            translate = (int(width * random.uniform(-0.05, 0.05)), int(height * random.uniform(-0.05, 0.05)))
            scale = random.uniform(0.9, 1.1)
            image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        image, mask = pad_if_needed(image, mask, self.image_size)
        mask_np = np.array(mask, dtype=np.uint8)
        top, left = random_crop_coordinates(mask_np, self.image_size, self.crop_foreground_prob)
        image = TF.crop(image, top, left, self.image_size, self.image_size)
        mask = TF.crop(mask, top, left, self.image_size, self.image_size)

        if random.random() < self.color_jitter_prob:
            image = self.color_jitter(image)

        if random.random() < self.blur_prob:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.2)))

        if random.random() < self.jpeg_prob:
            image = jpeg_compress(image, quality=random.randint(55, 90))

        return image, mask

    def add_tensor_noise(self, image_t: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.noise_prob:
            return image_t
        noise = torch.randn_like(image_t) * random.uniform(0.01, 0.04)
        return (image_t + noise).clamp_(0.0, 1.0)


class CarPartSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: Dataset,
        image_size: int,
        train: bool,
        hflip_prob: float,
        scale_min: float = 0.75,
        scale_max: float = 1.50,
        affine_prob: float = 0.7,
        crop_foreground_prob: float = 0.8,
        color_jitter_prob: float = 0.8,
        blur_prob: float = 0.15,
        jpeg_prob: float = 0.10,
        noise_prob: float = 0.15,
    ) -> None:
        self.split = split
        self.image_size = image_size
        self.train = train
        self.augmenter = (
            SegmentationTrainAugmenter(
                image_size=image_size,
                hflip_prob=hflip_prob,
                scale_min=scale_min,
                scale_max=scale_max,
                affine_prob=affine_prob,
                crop_foreground_prob=crop_foreground_prob,
                color_jitter_prob=color_jitter_prob,
                blur_prob=blur_prob,
                jpeg_prob=jpeg_prob,
                noise_prob=noise_prob,
            )
            if train
            else None
        )

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        example = self.split[idx]
        image = example["image"].convert("RGB")
        mask = build_segmentation_mask(example)

        if self.train and self.augmenter is not None:
            image, mask = self.augmenter(image, mask)
        else:
            image, mask = resize_and_pad(image, mask, self.image_size)

        image_t = TF.to_tensor(image)
        if self.train and self.augmenter is not None:
            image_t = self.augmenter.add_tensor_noise(image_t)
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


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


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
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    data_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    num_batches = len(loader)
    progress = tqdm(
        loader,
        total=num_batches,
        desc=f"train {epoch:02d}",
        dynamic_ncols=True,
        leave=False,
    )
    end = time.perf_counter()

    for step, (images, masks) in enumerate(progress, start=1):
        data_time = time.perf_counter() - end
        data_time_meter.update(data_time)
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

        batch_time = time.perf_counter() - end
        batch_time_meter.update(batch_time)
        batch_size = int(images.shape[0])
        loss_meter.update(loss.item(), n=batch_size)

        if step % print_freq == 0 or step == num_batches:
            progress.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                data=f"{data_time_meter.avg:.3f}s",
                batch=f"{batch_time_meter.avg:.3f}s",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )
        end = time.perf_counter()

    progress.close()
    return {
        "loss": loss_meter.avg,
        "data_time": data_time_meter.avg,
        "batch_time": batch_time_meter.avg,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    epoch: int,
    print_freq: int,
) -> Dict[str, object]:
    model.eval()
    loss_meter = AverageMeter()
    data_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64, device=device)
    num_batches = len(loader)
    progress = tqdm(
        loader,
        total=num_batches,
        desc=f"val   {epoch:02d}",
        dynamic_ncols=True,
        leave=False,
    )
    end = time.perf_counter()

    for step, (images, masks) in enumerate(progress, start=1):
        data_time = time.perf_counter() - end
        data_time_meter.update(data_time)
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

        batch_time = time.perf_counter() - end
        batch_time_meter.update(batch_time)
        batch_size = int(images.shape[0])
        loss_meter.update(loss.item(), n=batch_size)
        update_confusion_matrix(confusion, preds, masks, NUM_CLASSES)
        if step % max(1, print_freq) == 0 or step == num_batches:
            progress.set_postfix(
                loss=f"{loss_meter.avg:.4f}",
                data=f"{data_time_meter.avg:.3f}s",
                batch=f"{batch_time_meter.avg:.3f}s",
            )
        end = time.perf_counter()

    progress.close()
    metrics = compute_metrics(confusion.cpu())
    metrics["loss"] = loss_meter.avg
    metrics["data_time"] = data_time_meter.avg
    metrics["batch_time"] = batch_time_meter.avg
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
    if args.scale_min <= 0 or args.scale_max <= 0 or args.scale_min > args.scale_max:
        raise ValueError("scale-min and scale-max must be positive and satisfy scale-min <= scale-max")
    probability_args = {
        "hflip-prob": args.hflip_prob,
        "affine-prob": args.affine_prob,
        "crop-foreground-prob": args.crop_foreground_prob,
        "color-jitter-prob": args.color_jitter_prob,
        "blur-prob": args.blur_prob,
        "jpeg-prob": args.jpeg_prob,
        "noise-prob": args.noise_prob,
    }
    for name, value in probability_args.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

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
            "train_indices_by_source": {
                "train": list(range(len(official_train_split))),
                "test": [],
            },
            "val_indices_by_source": {
                "train": [],
                "test": list(range(len(official_test_split))),
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
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        affine_prob=args.affine_prob,
        crop_foreground_prob=args.crop_foreground_prob,
        color_jitter_prob=args.color_jitter_prob,
        blur_prob=args.blur_prob,
        jpeg_prob=args.jpeg_prob,
        noise_prob=args.noise_prob,
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
    best_epoch = args.epochs
    history: List[Dict[str, object]] = []

    print(
        f"training on device={device} amp={use_amp} encoder={args.encoder_name} image_size={args.image_size}"
    )
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
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
        train_loss = train_stats["loss"]

        epoch_metrics: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_data_time": train_stats["data_time"],
            "train_batch_time": train_stats["batch_time"],
            "lr": optimizer.param_groups[0]["lr"],
        }

        if epoch % args.eval_every == 0:
            eval_metrics = evaluate(
                model,
                val_loader,
                criterion,
                device,
                use_amp=use_amp,
                epoch=epoch,
                print_freq=args.print_freq,
            )
            epoch_metrics.update(
                {
                    "val_loss": eval_metrics["loss"],
                    "val_data_time": eval_metrics["data_time"],
                    "val_batch_time": eval_metrics["batch_time"],
                    "pixel_accuracy": eval_metrics["pixel_accuracy"],
                    "mean_iou": eval_metrics["mean_iou"],
                    "foreground_mean_iou": eval_metrics["foreground_mean_iou"],
                }
            )
            print(
                "epoch={:02d} train_loss={:.4f} train_data={:.3f}s train_batch={:.3f}s "
                "val_loss={:.4f} val_data={:.3f}s val_batch={:.3f}s "
                "miou={:.4f} fg_miou={:.4f} pix_acc={:.4f}".format(
                    epoch,
                    train_loss,
                    train_stats["data_time"],
                    train_stats["batch_time"],
                    eval_metrics["loss"],
                    eval_metrics["data_time"],
                    eval_metrics["batch_time"],
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
                best_epoch = epoch
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
            print(
                "epoch={:02d} train_loss={:.4f} train_data={:.3f}s train_batch={:.3f}s".format(
                    epoch,
                    train_loss,
                    train_stats["data_time"],
                    train_stats["batch_time"],
                )
            )

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

    if args.refit_on_full_data:
        full_train_split = combine_splits([official_train_split, official_test_split])
        full_train_dataset = CarPartSegmentationDataset(
            full_train_split,
            image_size=args.image_size,
            train=True,
            hflip_prob=args.hflip_prob,
        )
        full_train_loader = DataLoader(
            full_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **common_loader_kwargs,
        )

        refit_model = build_model(args.encoder_name, encoder_weights=encoder_weights).to(device)
        refit_optimizer = AdamW(refit_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        refit_scheduler = CosineAnnealingLR(refit_optimizer, T_max=best_epoch, eta_min=args.min_lr)
        refit_scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

        print(f"refitting on all data for {best_epoch} epochs")
        for epoch in range(1, best_epoch + 1):
            refit_stats = train_one_epoch(
                model=refit_model,
                loader=full_train_loader,
                optimizer=refit_optimizer,
                criterion=criterion,
                device=device,
                epoch=epoch,
                print_freq=args.print_freq,
                use_amp=use_amp,
                scaler=refit_scaler,
            )
            print(
                "refit_epoch={:02d} train_loss={:.4f} train_data={:.3f}s train_batch={:.3f}s".format(
                    epoch,
                    refit_stats["loss"],
                    refit_stats["data_time"],
                    refit_stats["batch_time"],
                )
            )
            refit_scheduler.step()

        refit_path = output_dir / "model_refit_full.pt"
        torch.save(
            {
                "model_state_dict": refit_model.state_dict(),
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
                "refit_epochs": best_epoch,
                "selection_metric": args.best_metric,
                "split_summary": split_summary,
            },
            refit_path,
        )
        print(f"saved refit model: {refit_path}")

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
