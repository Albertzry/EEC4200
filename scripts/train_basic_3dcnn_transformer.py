#!/usr/bin/env python3
"""
Basic 3D CNN + Transformer training script for HMDB51-style frame folders.

This file is intentionally simple and beginner-friendly:
- hyperparameters are grouped at the top
- dataset, model, training and evaluation are in one script
- only keeps the minimum parts needed for training
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ============================================================
# Hyperparameters and easy-to-change settings
# 超参数都尽量放在前面，后面训练时你主要改这里。
# Put the hyperparameters near the top so they are easy to change later.
# ============================================================

NUM_CLASSES = 8
NUM_FRAMES = 16
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
INITIAL_LEARNING_RATE = 5e-4
FINAL_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-1
CNN_CHANNELS = [64, 128, 256]
TRANSFORMER_DIM = 64
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 3
TRANSFORMER_FF_DIM = 256
DROPOUT = 0.5
NUM_WORKERS = 12
DEVICE = "cuda" 

TRAIN_MANIFEST_PATH = "outputs/hmdb51_preprocessed/train_manifest.csv"
TEST_MANIFEST_PATH = "outputs/hmdb51_preprocessed/test_manifest.csv"
OUTPUT_DIR = "outputs/hmdb51_training_basic"

NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]


def parse_args() -> argparse.Namespace:
    # 命令行参数用于覆盖默认路径。
    # Command line arguments let you override the default paths.
    parser = argparse.ArgumentParser(
        description="Train a basic 3D CNN + Transformer on preprocessed HMDB51 frames."
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
        help="Directory for logs, predictions and model weights.",
    )
    return parser.parse_args()
class HMDB51FrameDataset(Dataset):
    # 这个 Dataset 直接读取 manifest，再去 frame_dir 里面拿 32 张图片。
    # This Dataset reads the manifest first, then loads 32 images from each frame folder.
    def __init__(
        self,
        manifest_path: str,
        num_frames: int = NUM_FRAMES,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.manifest_path = manifest_path
        self.num_frames = num_frames
        self.image_size = image_size
        self.data = pd.read_csv(manifest_path)

    def __len__(self) -> int:
        return len(self.data)

    def _load_one_frame(self, frame_path: Path) -> torch.Tensor:
        # 读取单张图片，并转换成 PyTorch 需要的张量格式。
        # Read one image and convert it into the tensor format used by PyTorch.
        image = cv2.imread(str(frame_path))
        if image is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = cv2.resize(image, (self.image_size, self.image_size))

        image = image.astype(np.float32) / 255.0
        image = (image - np.array(NORMALIZE_MEAN, dtype=np.float32)) / np.array(
            NORMALIZE_STD, dtype=np.float32
        )
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image)

    def _load_video_frames(self, frame_dir: str) -> torch.Tensor:
        # 按顺序读取 32 帧，最后组织成 [C, T, H, W]。
        # Read 32 frames in time order and return a tensor with shape [C, T, H, W].
        frame_folder = Path(frame_dir)
        frames: list[torch.Tensor] = []
        for frame_index in range(self.num_frames):
            frame_path = frame_folder / f"frame_{frame_index:03d}.jpg"
            frames.append(self._load_one_frame(frame_path))

        video = torch.stack(frames, dim=1)
        return video.float()

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data.iloc[index]
        video = self._load_video_frames(row["frame_dir"])
        label = torch.tensor(int(row["label"]), dtype=torch.long)

        return {
            "video": video,
            "label": label,
            "sample_name": str(row["sample_name"]),
            "frame_dir": str(row["frame_dir"]),
        }


class BasicPositionalEncoding(nn.Module):
    # Transformer 不知道时间顺序，所以这里加一个最基础的位置编码。
    # The Transformer does not know time order by itself, so we add a simple positional encoding.
    def __init__(self, max_length: int, embed_dim: int) -> None:
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        return x + self.position_embedding[:, :length, :]


class Basic3DCNNTransformer(nn.Module):
    # 模型思路：
    # 1. 用 3D CNN 提取时空特征
    # 2. 把每个时间步的特征送给 Transformer
    # 3. 用分类头输出 8 类结果
    #
    # Model idea:
    # 1. Use 3D CNN to extract spatio-temporal features
    # 2. Feed the time-step features into a Transformer
    # 3. Use a classifier head to predict the 8 classes
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

        # 这里故意只写 3 个最基础的 3D 卷积块，方便初学者理解。
        # We intentionally use only 3 simple 3D convolution blocks for readability.
        #
        # 输入一开始是 [B, 3, T, H, W]
        # At the beginning, the input shape is [B, 3, T, H, W]
        #
        # B: batch size
        # 3: RGB 三个通道 / 3 RGB channels
        # T: 时间维，也就是 32 帧 / time dimension, here 32 frames
        # H, W: 图像高和宽 / image height and width
        #
        # 这里的池化只在空间维上缩小图像大小，不压缩时间维。
        # The pooling here only reduces spatial size, not the time dimension.
        #
        # 这样做的原因是：
        # 1. 3D CNN 先学习“短时间范围内”的局部时空模式
        # 2. 后面 Transformer 再去看整个时间序列
        #
        # Why do this:
        # 1. 3D CNN first learns local spatio-temporal patterns
        # 2. Then the Transformer looks at the whole temporal sequence
        self.features = nn.Sequential(
            nn.Conv3d(3, cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
        )

        # 经过 3D CNN 后，通道数会变成 cnn_channels[-1]，也就是 64。
        # After the 3D CNN, the channel size becomes cnn_channels[-1], which is 64 here.
        #
        # 接着我们会把空间维做平均池化，只留下每个时间步的特征向量。
        # Then we average over spatial dimensions and keep one feature vector per time step.
        #
        # 如果前面得到 [B, 64, T, H', W']，
        # 空间平均后就会变成 [B, 64, T]。
        # If the feature map is [B, 64, T, H', W'],
        # after spatial averaging it becomes [B, 64, T].
        #
        # Transformer 更喜欢处理 [B, T, C]，
        # 所以后面还会转成 [B, T, 64]。
        # Transformer prefers [B, T, C],
        # so later we will convert it into [B, T, 64].
        self.feature_projection = nn.Linear(cnn_channels[-1], transformer_dim)
        self.position_encoding = BasicPositionalEncoding(num_frames, transformer_dim)

        # 这里使用最基础的 TransformerEncoder。
        # We use the most basic TransformerEncoder here.
        #
        # d_model = 每个时间步特征向量的长度
        # d_model = feature size of each time step
        #
        # nhead = 多头注意力里的头数
        # nhead = number of attention heads
        #
        # num_layers = Transformer 堆叠多少层
        # num_layers = how many Transformer layers we stack
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
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(transformer_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状: [B, 3, 32, 224, 224]
        # Input shape: [B, 3, 32, 224, 224]
        x = self.features(x)

        # 现在 x 的形状大致是 [B, 64, 32, 28, 28]
        # 这里的 28, 28 只是一个例子，表示空间尺寸已经被池化缩小了。
        # Now x is roughly [B, 64, 32, 28, 28]
        # Here 28, 28 is just an example showing that the spatial size is reduced.

        # 对空间维做平均池化，保留时间维。
        # Average over the spatial dimensions and keep the time dimension.
        x = x.mean(dim=4).mean(dim=3)

        # 现在变成 [B, 64, 32]
        # Now it becomes [B, 64, 32]

        # 变成 [B, T, C]，这样 Transformer 更容易处理。
        # Convert to [B, T, C], which is easier for the Transformer to use.
        x = x.permute(0, 2, 1)

        # 现在变成 [B, 32, 64]
        # Now it becomes [B, 32, 64]

        # 线性层把特征投影到 Transformer 的维度。
        # The linear layer projects the features to the Transformer dimension.
        x = self.feature_projection(x)

        # 加位置编码，让模型知道第 1 帧、第 2 帧……第 32 帧的顺序。
        # Add positional encoding so the model knows the order of frame 1, 2, ... 32.
        x = self.position_encoding(x)

        # Transformer 负责建模“整个时间序列”的关系。
        # The Transformer models the relationship across the whole time sequence.
        x = self.transformer(x)

        # 对所有时间步做平均，得到整个视频的一个全局表示。
        # Average all time steps to get one global representation of the whole video.
        x = x.mean(dim=1)
        x = self.dropout(x)

        # 最后用全连接层输出 8 个类别的分数。
        # Finally, use a fully connected layer to output scores for 8 classes.
        logits = self.classifier(x)
        return logits


def build_dataloader(manifest_path: str, shuffle: bool) -> DataLoader:
    # DataLoader 负责按 batch 把数据送进模型。
    # The DataLoader groups samples into batches for the model.
    dataset = HMDB51FrameDataset(manifest_path=manifest_path)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
def run_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
) -> tuple[float, float]:
    # 如果 optimizer 不为空，就做训练；否则只做验证。
    # If optimizer is provided, we train; otherwise, we only evaluate.
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, leave=False):
        videos = batch["video"].to(device)
        labels = batch["label"].to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(videos)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (torch.argmax(logits, dim=1) == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples if total_samples > 0 else 0.0
    average_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return average_loss, average_accuracy


def plot_training_curves(history: list[dict[str, Any]], output_dir: Path) -> None:
    # 每个 epoch 后都重画一次曲线图，方便观察训练过程。
    # Re-draw the curve after each epoch so you can monitor the training process.
    if not history:
        return

    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    test_losses = [row["test_loss"] for row in history]
    train_accuracies = [row["train_accuracy"] for row in history]
    test_accuracies = [row["test_accuracy"] for row in history]
    learning_rates = [row["learning_rate"] for row in history]

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, test_losses, marker="o", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_accuracies, marker="o", label="Train Accuracy")
    plt.plot(epochs, test_accuracies, marker="o", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
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


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            videos = batch["video"].to(device)
            labels = batch["label"].to(device)
            logits = model(videos)
            loss = criterion(logits, labels)

            predictions = torch.argmax(logits, dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples if total_samples > 0 else 0.0
    average_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return average_loss, average_accuracy


def save_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    # 保存每个 epoch 的数值，便于后续查表或画图。
    # Save the numeric results of each epoch for later analysis or plotting.
    history_path = output_dir / "history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)


def main() -> None:
    # main 函数把整个流程串起来：读数据、建模型、训练、评估、保存结果。
    # The main function connects the full pipeline: load data, build model, train, evaluate, save.
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("===== Basic 3D CNN + Transformer Training =====")
    print(f"Device: {DEVICE}")
    print(f"Train manifest: {args.train_manifest}")
    print(f"Test manifest: {args.test_manifest}")
    print(f"Output dir: {output_dir}")

    train_loader = build_dataloader(args.train_manifest, shuffle=True)
    test_loader = build_dataloader(args.test_manifest, shuffle=False)

    model = Basic3DCNNTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    # 使用 AdamW: 这是专门为带有 L2 正则化(Weight Decay) 修复过后的 Adam 优化器，能更有效地防止过拟合
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=INITIAL_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY, # 这就是 L2 正则化项系数
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=FINAL_LEARNING_RATE / INITIAL_LEARNING_RATE,
        total_iters=EPOCHS,
    )

    history: list[dict[str, Any]] = []
    best_accuracy = -1.0
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{EPOCHS} | learning_rate={current_lr:.6f}")

        train_loss, train_accuracy = run_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=DEVICE,
        )

        test_loss, test_accuracy = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=DEVICE,
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_accuracy, 6),
            "test_loss": round(test_loss, 6),
            "test_accuracy": round(test_accuracy, 6),
            "learning_rate": current_lr,
        }
        history.append(epoch_result)
        save_history(history, output_dir)
        plot_training_curves(history, output_dir)

        print(
            f"train_loss={train_loss:.4f}, "
            f"train_accuracy={train_accuracy:.4f}, "
            f"test_loss={test_loss:.4f}, "
            f"test_accuracy={test_accuracy:.4f}, "
            f"learning_rate={current_lr:.6f}"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pth")

        scheduler.step()

    save_history(history, output_dir)
    plot_training_curves(history, output_dir)

    print("\n===== Training Finished =====")
    print(f"Best test accuracy: {best_accuracy:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best model saved to: {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
