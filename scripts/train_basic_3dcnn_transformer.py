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
from torchvision.models.video import r3d_18
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
UNFREEZE_LEARNING_RATE = 3e-5
FINAL_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 5e-3
NUM_FOLDS = 5

R3D_FEATURE_DIM = 512
TRANSFORMER_DIM = 256
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 2
TRANSFORMER_FF_DIM = 512
DROPOUT = 0.4

# ============================================================
# Freeze / unfreeze settings for the R3D backbone
# 先冻结前面的 R3D 层，过几轮再解冻。
# Freeze early R3D layers first, then unfreeze them after a few epochs.
# ============================================================
FREEZE_R3D_STEM = True
FREEZE_R3D_LAYER1 = True
FREEZE_R3D_LAYER2 = False
UNFREEZE_AT_EPOCH = 8

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


class BasicR3DTransformer(nn.Module):
    # 模型主思路：
    # 1. 输入是一个视频张量 [B, 3, T, H, W]
    # 2. 先用最基础的 R3D-18 backbone 提取更成熟的时空特征
    # 3. 再把这些时间特征送进 Transformer
    # 4. 最后用全连接层做分类
    #
    # Main idea:
    # 1. Input is a video tensor [B, 3, T, H, W]
    # 2. Use a basic R3D-18 backbone to extract stronger spatio-temporal features
    # 3. Feed the time sequence into a Transformer
    # 4. Use a linear layer for final classification
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        r3d_feature_dim: int = R3D_FEATURE_DIM,
        transformer_dim: int = TRANSFORMER_DIM,
        transformer_heads: int = TRANSFORMER_HEADS,
        transformer_layers: int = TRANSFORMER_LAYERS,
        transformer_ff_dim: int = TRANSFORMER_FF_DIM,
        dropout: float = DROPOUT,
        num_frames: int = NUM_FRAMES,
    ) -> None:
        super().__init__()

        # 第一部分：R3D backbone
        # First part: R3D backbone
        #
        # 这里不直接用 r3d_18 的最终分类层，而是只拿它前面的特征提取部分。
        # We do not use the final classification layer of r3d_18 here.
        # We only keep the feature extraction part.
        #
        # 这样做的原因是：
        # 1. R3D 是更成熟的 3D CNN backbone
        # 2. 它比手写的几层 Conv3d 更容易提取有效时空特征
        # 3. 我们后面还要接 Transformer，所以这里只需要特征，不要最终分类结果
        #
        # Why:
        # 1. R3D is a more mature 3D CNN backbone
        # 2. It usually extracts better video features than a few hand-written Conv3d layers
        # 3. We still want to use a Transformer later, so we only need backbone features here
        backbone = r3d_18(weights=None)
        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 第二部分：把 R3D 的输出变成 Transformer 能读的序列
        # Second part: turn the R3D output into a sequence for the Transformer
        #
        # 经过 R3D 后，张量仍然是 5 维：[B, C, T', H', W']
        # After R3D, the tensor is still 5D: [B, C, T', H', W']
        #
        # 我们先对空间维 H' 和 W' 求平均，只保留时间维 T'
        # Then we average over H' and W' to keep only the temporal sequence T'
        #
        # 这样每个时间步都会得到一个长度为 C 的特征向量
        # This gives one C-dimensional feature vector for each time step
        self.feature_projection = nn.Linear(r3d_feature_dim, transformer_dim)
        self.position_encoding = BasicPositionalEncoding(num_frames, transformer_dim)

        # 第三部分：Transformer encoder
        # Third part: Transformer encoder
        #
        # 这里保留你原来的思路：R3D 先提局部到中层时空特征，
        # Transformer 再看更长范围的时间关系。
        #
        # We keep your original idea here:
        # R3D extracts local / mid-level spatio-temporal features first,
        # then the Transformer models longer temporal relationships.
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

        # 最后是分类头
        # Final classification head
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(transformer_dim)
        self.classifier = nn.Linear(transformer_dim, num_classes)

        # 训练开始时，先冻结一部分早期 R3D 层。
        # At the start of training, freeze some early R3D layers first.
        self.freeze_early_r3d_layers()

    def freeze_early_r3d_layers(self) -> None:
        # 这个函数只负责把前面几层设成不更新。
        # This function only marks early layers as frozen.
        if FREEZE_R3D_STEM:
            for param in self.stem.parameters():
                param.requires_grad = False
        if FREEZE_R3D_LAYER1:
            for param in self.layer1.parameters():
                param.requires_grad = False
        if FREEZE_R3D_LAYER2:
            for param in self.layer2.parameters():
                param.requires_grad = False

    def unfreeze_all_r3d_layers(self) -> None:
        # 到了指定 epoch 后，把整个 R3D backbone 解冻。
        # After the chosen epoch, unfreeze the whole R3D backbone.
        for module in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in module.parameters():
                param.requires_grad = True

    def count_trainable_parameters(self) -> int:
        # 方便在日志里查看当前有多少参数在训练。
        # Useful for logging how many parameters are currently trainable.
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: [B, 3, 32, 224, 224]
        # Input: [B, 3, 32, 224, 224]

        # R3D 的特征提取部分
        # R3D feature extraction part
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 做空间平均池化，保留时间维
        # Spatial average pooling while keeping the time dimension
        #
        # [B, C, T', H', W'] -> [B, C, T']
        x = x.mean(dim=4).mean(dim=3)

        # [B, C, T'] -> [B, T', C]
        # Transformer 更喜欢 [batch, time, feature] 这样的格式
        # Transformer prefers [batch, time, feature]
        x = x.permute(0, 2, 1)

        # 把 R3D 特征映射到 Transformer 的特征维度
        # Project R3D features into the Transformer feature dimension
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


def build_model() -> nn.Module:
    # 单独写一个 build_model，后面训练和测试都复用。
    # Keep model creation in one place so both training and testing reuse it.
    return BasicR3DTransformer().to(DEVICE)


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    # 手动设置 optimizer 的学习率，避免引入更复杂的 scheduler。
    # Manually set optimizer learning rate to keep things simple.
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_epoch_learning_rate(epoch: int) -> float:
    # 两阶段学习率：
    # 1. 冻结阶段：使用较大的学习率训练后面的层
    # 2. 解冻之后：切到更小的学习率，再慢慢下降到 FINAL_LEARNING_RATE
    #
    # Two-stage learning rate:
    # 1. Frozen stage: use a larger LR for the later layers
    # 2. After unfreezing: switch to a smaller LR, then decay toward FINAL_LEARNING_RATE
    if epoch < UNFREEZE_AT_EPOCH:
        return LEARNING_RATE

    remaining_epochs = max(EPOCHS - UNFREEZE_AT_EPOCH, 1)
    progress = min(max(epoch - UNFREEZE_AT_EPOCH, 0), remaining_epochs) / remaining_epochs
    return UNFREEZE_LEARNING_RATE + (FINAL_LEARNING_RATE - UNFREEZE_LEARNING_RATE) * progress


def maybe_unfreeze_backbone(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    # 训练到指定 epoch 时，把前面冻结的 R3D 层解冻。
    # When training reaches the chosen epoch, unfreeze the frozen R3D layers.
    if epoch != UNFREEZE_AT_EPOCH:
        return

    if isinstance(model, BasicR3DTransformer):
        model.unfreeze_all_r3d_layers()
        set_optimizer_learning_rate(optimizer, UNFREEZE_LEARNING_RATE)
        print(
            f"Unfreezing R3D backbone at epoch {epoch}. "
            f"Trainable parameters: {model.count_trainable_parameters():,}. "
            f"New learning rate: {UNFREEZE_LEARNING_RATE:.6f}"
        )


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

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history: list[dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_epoch = -1

    print(f"\n===== Fold {fold_index + 1}/{NUM_FOLDS} =====")
    print(f"Train samples: {len(train_part)}")
    print(f"Validation samples: {len(val_part)}")
    if isinstance(model, BasicR3DTransformer):
        print(f"Initial trainable parameters: {model.count_trainable_parameters():,}")

    for epoch in range(1, EPOCHS + 1):
        maybe_unfreeze_backbone(model, optimizer, epoch)
        current_lr = get_epoch_learning_rate(epoch)
        set_optimizer_learning_rate(optimizer, current_lr)
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

    model = build_model()
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

    print("===== Basic R3D + Transformer 5-Fold Training =====")
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
