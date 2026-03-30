#!/usr/bin/env python3
"""
Basic 3D CNN + Transformer training script for HMDB51-style frame folders.

This version keeps the training pipeline simple:
- 5-fold cross validation on the training manifest
- train/validation curves only during training
- test set is used only once after training
- separate test mode is kept so you can run testing again later
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================
# Hyperparameters and easy-to-change settings
# 超参数放在最前面，方便你直接修改。
# Put hyperparameters at the top so they are easy to change.
# ============================================================

NUM_CLASSES = 8
NUM_FRAMES = 32
IMAGE_SIZE = 224
BATCH_SIZE = 12
EPOCHS = 50
LEARNING_RATE = 1e-4
FINAL_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 5e-3
NUM_FOLDS = 5

CNN_CHANNELS = [64, 128, 256, 512]
TRANSFORMER_DIM = 256
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 3
TRANSFORMER_FF_DIM = 512
DROPOUT = 0.5

NUM_WORKERS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_MANIFEST_PATH = "outputs/arid_preprocessed/train_manifest.csv"
TEST_MANIFEST_PATH = "outputs/arid_preprocessed/test_manifest.csv"
OUTPUT_DIR = "outputs/arid_training_basic"

NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Basic 3D CNN + Transformer with 5-fold validation."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="train: run 5-fold training, test: load one model and run test only.",
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default=TRAIN_MANIFEST_PATH,
        help="Path to train_manifest.csv.",
    )
    parser.add_argument(
        "--test-manifest",
        type=str,
        default=TEST_MANIFEST_PATH,
        help="Path to test_manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory for model weights, csv logs and plots.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Used in test mode. Path to a saved .pth model.",
    )
    return parser.parse_args()


class HMDB51FrameDataset(Dataset):
    # 这个 Dataset 只做一件事：读取 manifest，然后按 frame_dir 读图片。
    # This Dataset only does one job: read the manifest and then load frames from frame_dir.
    def __init__(
        self,
        dataframe: pd.DataFrame,
        num_frames: int = NUM_FRAMES,
        image_size: int = IMAGE_SIZE,
        is_training: bool = False,
    ) -> None:
        self.data = dataframe.reset_index(drop=True).copy()
        self.num_frames = num_frames
        self.image_size = image_size
        self.is_training = is_training
        
        # ColorJitter for dark video augmentation
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3) if is_training else None

    def __len__(self) -> int:
        return len(self.data)

    def _load_one_frame(self, frame_path: Path) -> torch.Tensor:
        image = cv2.imread(str(frame_path))
        if image is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size))

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(image)
        
        if self.color_jitter:
            tensor = self.color_jitter(tensor)
            
        mean_tensor = torch.tensor(NORMALIZE_MEAN, dtype=torch.float32).view(-1, 1, 1)
        std_tensor = torch.tensor(NORMALIZE_STD, dtype=torch.float32).view(-1, 1, 1)
        tensor = (tensor - mean_tensor) / std_tensor
        return tensor

    def _load_video_frames(self, frame_dir: str) -> torch.Tensor:
        # 最后输出 [C, T, H, W]。
        # The final output shape is [C, T, H, W].
        frame_folder = Path(frame_dir)
        frames: list[torch.Tensor] = []
        for frame_index in range(self.num_frames):
            frame_path = frame_folder / f"frame_{frame_index:03d}.jpg"
            frames.append(self._load_one_frame(frame_path))
        return torch.stack(frames, dim=1).float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data.iloc[index]
        return {
            "video": self._load_video_frames(str(row["frame_dir"])),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_name": str(row["sample_name"]),
            "relative_path": str(row["relative_path"]),
        }


class BasicPositionalEncoding(nn.Module):
    # Transformer 自己不知道第几帧在前、第几帧在后，所以这里加位置编码。
    # The Transformer does not know frame order by itself, so we add positional encoding.
    def __init__(self, max_length: int, embed_dim: int) -> None:
        super().__init__()
        self.position_embedding = nn.Parameter(torch.randn(1, max_length, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_embedding[:, : x.size(1), :]


class Basic3DCNNTransformer(nn.Module):
    # 模型主思路：
    # 1. 输入是一个视频张量 [B, 3, T, H, W]
    # 2. 3D CNN 先提取局部时空特征
    # 3. Transformer 再建模整段视频的时间关系
    # 4. 最后用全连接层做分类
    #
    # Main idea:
    # 1. Input is a video tensor [B, 3, T, H, W]
    # 2. 3D CNN extracts local spatio-temporal features first
    # 3. Transformer models the full temporal relationship
    # 4. A linear layer does the final classification
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        cnn_channels: list[int] | None = None,
        transformer_dim: int = TRANSFORMER_DIM,
        transformer_heads: int = TRANSFORMER_HEADS,
        transformer_layers: int = TRANSFORMER_LAYERS,
        transformer_ff_dim: int = TRANSFORMER_FF_DIM,
        dropout: float = DROPOUT,
        num_frames: int = NUM_FRAMES,
    ) -> None:
        super().__init__()

        if cnn_channels is None:
            cnn_channels = CNN_CHANNELS

        # 3D CNN 
        self.features = nn.Sequential(
            nn.Conv3d(3, cnn_channels[0], kernel_size=3, padding=1),  # Input [B, 3, T, H, W]
            nn.BatchNorm3d(cnn_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),                     # keep time
            nn.Conv3d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(cnn_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(cnn_channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(cnn_channels[2], cnn_channels[3], kernel_size=3, padding=1),
            nn.BatchNorm3d(cnn_channels[3]),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )

        self.feature_projection = nn.Linear(cnn_channels[-1], transformer_dim)  # Map to transformer input dim
        self.position_encoding = BasicPositionalEncoding(num_frames, transformer_dim)

        # Transformer encoder 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        # Classification output layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(transformer_dim)
        self.classifier = nn.Linear(transformer_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: [B, 3, 32, 224, 224]
        # Input: [B, 3, 32, 224, 224]
        x = self.features(x)

        # 做空间平均池化，保留时间维
        # Spatial average pooling while keeping the time dimension
        x = x.mean(dim=4).mean(dim=3)

        # [B, C, T] -> [B, T, C]
        # This is the shape expected by the Transformer
        x = x.permute(0, 2, 1)

        # 把 CNN 特征映射到 Transformer 的特征维度
        # Project CNN features into the Transformer feature dimension
        x = self.feature_projection(x)

        # 加入位置信息，让模型知道帧顺序
        # Add position information so the model knows frame order
        x = self.position_encoding(x)

        # Transformer 处理整个时间序列
        # Transformer processes the full time sequence
        x = self.transformer(x)

        # 对时间维求平均，得到整段视频的表示
        # Average over time to get one representation for the whole video
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.layer_norm(x)

        # 输出 8 类分数
        # Output logits for 8 classes
        return self.classifier(x)


def build_dataloader(dataframe: pd.DataFrame, shuffle: bool) -> DataLoader:
    dataset = HMDB51FrameDataset(dataframe=dataframe, is_training=shuffle)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> tuple[float, float]:
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, leave=False):
        videos = batch["video"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(videos)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    return run_one_epoch(model, dataloader, criterion, optimizer=None, device=device)


def plot_training_curves(history: list[dict[str, Any]], output_dir: Path) -> None:
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]
    train_accuracies = [row["train_accuracy"] for row in history]
    val_accuracies = [row["val_accuracy"] for row in history]
    learning_rates = [row["learning_rate"] for row in history]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(epochs, val_accuracies, marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train / Validation Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, learning_rates, marker="o", color="green", label="Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.title("Learning Rate Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close()


def save_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)


def build_grouped_folds(train_df: pd.DataFrame, num_folds: int) -> list[set[str]]:
    # 关键点：
    # 同一个原视频(relative_path)的原始样本和增强样本必须进同一折，
    # 否则会出现训练/验证泄漏。
    #
    # Key point:
    # The original sample and all augmented samples from the same relative_path
    # must stay in the same fold, otherwise train/validation leakage happens.
    folds: list[set[str]] = [set() for _ in range(num_folds)]

    grouped = (
        train_df[["relative_path", "label"]]
        .drop_duplicates()
        .sort_values(["label", "relative_path"])
    )

    for label_value in sorted(grouped["label"].unique()):
        class_paths = grouped[grouped["label"] == label_value]["relative_path"].tolist()
        shuffled_paths = list(np.random.permutation(class_paths))
        split_paths = np.array_split(shuffled_paths, num_folds)
        for fold_index, part in enumerate(split_paths):
            folds[fold_index].update(part.tolist())

    return folds


def train_one_fold(
    fold_index: int,
    train_part: pd.DataFrame,
    val_part: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    fold_dir = output_dir / f"fold_{fold_index + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(train_part, shuffle=True)
    val_loader = build_dataloader(val_part, shuffle=False)

    model = Basic3DCNNTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    # 计算线性衰减的最终系数
    # Calculate the end_factor for linear decay
    # start_factor = 1.0 (starts at LEARNING_RATE)
    # end_factor = FINAL_LEARNING_RATE / LEARNING_RATE
    end_factor = FINAL_LEARNING_RATE / LEARNING_RATE
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=end_factor,
        total_iters=EPOCHS,
    )

    history: list[dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_epoch = -1

    print(f"\n===== Fold {fold_index + 1}/{NUM_FOLDS} =====")
    print(f"Train samples: {len(train_part)}")
    print(f"Validation samples: {len(val_part)}")

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nFold {fold_index + 1} Epoch {epoch}/{EPOCHS} | lr={current_lr:.6f}")

        train_loss, train_accuracy = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
        )
        val_loss, val_accuracy = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=DEVICE,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_accuracy, 6),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_accuracy, 6),
                "learning_rate": current_lr,
            }
        )
        save_history(history, fold_dir)
        plot_training_curves(history, fold_dir)

        print(
            f"train_loss={train_loss:.4f}, "
            f"train_accuracy={train_accuracy:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_accuracy={val_accuracy:.4f}, "
            f"learning_rate={current_lr:.6f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), fold_dir / "best_model.pth")

        scheduler.step()

    return {
        "fold": fold_index + 1,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "model_path": str(fold_dir / "best_model.pth"),
        "history_path": str(fold_dir / "history.csv"),
    }


def run_test_only(
    model_path: str, test_manifest: str, output_dir: Path | None = None
) -> tuple[float, float]:
    test_df = pd.read_csv(test_manifest)
    class_mapping = test_df[["label", "class_name"]].drop_duplicates().sort_values("label")
    class_names = class_mapping["class_name"].tolist()

    test_loader = build_dataloader(test_df, shuffle=False)

    model = Basic3DCNNTransformer().to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    print(f"Running evaluation on {len(test_df)} samples...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            videos = batch["video"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = model(videos)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_df)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean()

    # Generate and print classification report
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    )
    print("\n===== Classification Report =====")
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    if output_dir:
        # Save metrics to text file
        with open(output_dir / "test_classification_report.txt", "w") as f:
            f.write(report)

        # Plot and save confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        cm_path = output_dir / "test_confusion_matrix.png"
        plt.savefig(cm_path, dpi=200)
        plt.close()
        print(f"Confusion matrix saved to {cm_path}")

        # Save results to Markdown
        md_content = f"""# HMDB51 Test Results Summary

## Performance Summary
- **Test Loss:** {avg_loss:.6f}
- **Test Accuracy:** {accuracy:.4f} ({accuracy * 100:.2f}%)

## Classification Report
```text
{report}
```

## Confusion Matrix
![Confusion Matrix]({cm_path.name})
"""
        with open(output_dir / "test_results_summary.md", "w") as f:
            f.write(md_content)
        print(f"Markdown summary saved to {output_dir / 'test_results_summary.md'}")

    return avg_loss, accuracy


def run_training(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_manifest)
    test_df = pd.read_csv(args.test_manifest)

    print("===== Basic 3D CNN + Transformer 5-Fold Training =====")
    print(f"Device: {DEVICE}")
    print(f"Train manifest: {args.train_manifest}")
    print(f"Test manifest: {args.test_manifest}")
    print(f"Output dir: {output_dir}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    folds = build_grouped_folds(train_df, NUM_FOLDS)
    fold_results: list[dict[str, Any]] = []

    for fold_index, val_relative_paths in enumerate(folds):
        val_mask = train_df["relative_path"].isin(val_relative_paths)
        val_part = train_df[val_mask].copy()
        train_part = train_df[~val_mask].copy()

        fold_result = train_one_fold(
            fold_index=fold_index,
            train_part=train_part,
            val_part=val_part,
            output_dir=output_dir,
        )
        fold_results.append(fold_result)

    fold_summary = pd.DataFrame(fold_results)
    fold_summary.to_csv(output_dir / "cross_validation_summary.csv", index=False)

    best_row = fold_summary.sort_values("best_val_accuracy", ascending=False).iloc[0]
    best_model_path = str(best_row["model_path"])

    print("\n===== Cross Validation Summary =====")
    print(fold_summary.to_string(index=False))
    print(f"\nBest fold model: {best_model_path}")

    # 训练全部完成后，test 只跑一次
    # After all folds finish, run test only once
    test_loss, test_accuracy = run_test_only(
        best_model_path, args.test_manifest, output_dir=output_dir
    )
    print("\n===== Final Test Result Summary =====")
    print(f"Best Fold: {best_row['fold']}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    pd.DataFrame(
        [
            {
                "best_fold": best_row["fold"],
                "best_model_path": best_model_path,
                "test_loss": round(test_loss, 6),
                "test_accuracy": round(test_accuracy, 6),
            }
        ]
    ).to_csv(output_dir / "final_test_result.csv", index=False)


def run_test_mode(args: argparse.Namespace) -> None:
    if not args.model_path:
        raise ValueError("In test mode, you must provide --model-path")

    print("===== Test Only Mode =====")
    print(f"Device: {DEVICE}")
    print(f"Model path: {args.model_path}")
    print(f"Test manifest: {args.test_manifest}")

    test_loss, test_accuracy = run_test_only(
        args.model_path, args.test_manifest, output_dir=Path(args.output_dir)
    )
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        run_training(args)
    else:
        run_test_mode(args)


if __name__ == "__main__":
    main()
