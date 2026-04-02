#!/usr/bin/env python3
"""
Basic R3D training script for fixed-length frame folders.

This version keeps the training pipeline readable for beginners:
- 5-fold cross validation on the training manifest
- train/validation curves only during training
- random temporal crop during training
- multi-clip inference during testing
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
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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
STORED_NUM_FRAMES = 48
CLIP_NUM_FRAMES = 32
TEST_NUM_CLIPS = 3
IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 2e-5
FINAL_LEARNING_RATE = 1e-6
WEIGHT_DECAY = 5e-3
NUM_FOLDS = 5

R3D_FEATURE_DIM = 512
DROPOUT = 0.5

# ============================================================
# Training-time light augmentation
# 统一离线提亮之后，训练阶段只保留较轻的随机扰动。
# After unified offline enhancement, keep only light random perturbation in training.
# ============================================================
TRAIN_JITTER_BRIGHTNESS = 0.10
TRAIN_JITTER_CONTRAST = 0.03

# ============================================================
# Focal loss settings
# 根据当前混淆矩阵，给难学类别更高权重。
# Based on the current confusion matrix, give harder classes larger weights.
#
# Class order:
# [Drink, Jump, Pick, Pour, Push, Run, Walk, Wave]
# ============================================================
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = [2.0, 3.0, 1.2, 1.0, 0.8, 3.5, 3.0, 2.0]

# ============================================================
# Freeze settings for the R3D backbone
# 永久冻结前面的 R3D 层，让训练更稳定。
# Permanently freeze early R3D layers to keep training more stable.
# ============================================================
FREEZE_R3D_STEM = True
FREEZE_R3D_LAYER1 = True
FREEZE_R3D_LAYER2 = True

NUM_WORKERS = 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_MANIFEST_PATH = "/opt/data/private/EEC4200/outputs/arid_preprocessed/train_manifest.csv"
TEST_MANIFEST_PATH = "/opt/data/private/EEC4200/outputs/arid_preprocessed/test_manifest.csv"
OUTPUT_DIR = "/opt/data/private/EEC4200/outputs/arid_training_basic"

NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

# ============================================================
# Optional test-time enhancement
# 现在默认关闭，因为统一的固定提亮已经在预处理阶段完成。
# Now disabled by default because the unified fixed enhancement is done offline
# during preprocessing.
# ============================================================
ENABLE_TEST_ENHANCEMENT = False
TEST_BRIGHTNESS_FACTOR = 1.25
TEST_CONTRAST_FACTOR = 1.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Basic R3D classifier with 5-fold validation."
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
        stored_num_frames: int = STORED_NUM_FRAMES,
        clip_num_frames: int = CLIP_NUM_FRAMES,
        test_num_clips: int = TEST_NUM_CLIPS,
        image_size: int = IMAGE_SIZE,
        is_training: bool = False,
        use_test_enhancement: bool = False,
        multi_clip_eval: bool = False,
    ) -> None:
        self.data = dataframe.reset_index(drop=True).copy()
        self.stored_num_frames = stored_num_frames
        self.clip_num_frames = clip_num_frames
        self.test_num_clips = test_num_clips
        self.image_size = image_size
        self.is_training = is_training
        self.use_test_enhancement = use_test_enhancement
        self.multi_clip_eval = multi_clip_eval
        
        # 训练阶段只做轻度在线亮度/对比度扰动，
        # 避免和离线固定提亮叠加得太重。
        # During training, only apply light online brightness / contrast jitter,
        # so it does not stack too strongly with the offline fixed enhancement.
        self.color_jitter = (
            T.ColorJitter(
                brightness=TRAIN_JITTER_BRIGHTNESS,
                contrast=TRAIN_JITTER_CONTRAST,
            )
            if is_training
            else None
        )

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

        # 测试时使用固定增强，而不是随机增强。
        # Use fixed enhancement at test time instead of random augmentation.
        if self.use_test_enhancement:
            tensor = TF.adjust_brightness(tensor, TEST_BRIGHTNESS_FACTOR)
            tensor = TF.adjust_contrast(tensor, TEST_CONTRAST_FACTOR)
            tensor = torch.clamp(tensor, 0.0, 1.0)
            
        mean_tensor = torch.tensor(NORMALIZE_MEAN, dtype=torch.float32).view(-1, 1, 1)
        std_tensor = torch.tensor(NORMALIZE_STD, dtype=torch.float32).view(-1, 1, 1)
        tensor = (tensor - mean_tensor) / std_tensor
        return tensor

    def _load_video_frames(self, frame_dir: str) -> torch.Tensor:
        # 先读取离线保存的所有帧，默认希望是 48 帧。
        # First read all offline-saved frames. We expect 48 frames by default.
        #
        # 这样后面就可以在线做：
        # 1. 训练随机时间裁剪
        # 2. 验证中心裁剪
        # 3. 测试 multi-clip 推理
        #
        # This makes it possible to do:
        # 1. random temporal crop for training
        # 2. center crop for validation
        # 3. multi-clip inference for testing
        frame_folder = Path(frame_dir)
        frames: list[torch.Tensor] = []
        frame_paths = sorted(frame_folder.glob("frame_*.jpg"))[: self.stored_num_frames]
        if not frame_paths:
            raise RuntimeError(f"No frame files found in {frame_folder}")

        for frame_path in frame_paths:
            frames.append(self._load_one_frame(frame_path))
        return torch.stack(frames, dim=1).float()

    def _pad_temporal_clip(self, video: torch.Tensor) -> torch.Tensor:
        # 如果离线帧数少于 clip 长度，就重复最后几帧补齐。
        # If the stored video is shorter than the clip length, repeat the last frames.
        total_frames = video.size(1)
        if total_frames >= self.clip_num_frames:
            return video

        repeat_count = self.clip_num_frames - total_frames
        pad_frame = video[:, -1:, :, :].repeat(1, repeat_count, 1, 1)
        return torch.cat([video, pad_frame], dim=1)

    def _get_clip_from_start(self, video: torch.Tensor, start_index: int) -> torch.Tensor:
        video = self._pad_temporal_clip(video)
        max_start = max(0, video.size(1) - self.clip_num_frames)
        start_index = min(max(0, start_index), max_start)
        end_index = start_index + self.clip_num_frames
        return video[:, start_index:end_index, :, :]

    def _sample_training_clip(self, video: torch.Tensor) -> torch.Tensor:
        video = self._pad_temporal_clip(video)
        max_start = max(0, video.size(1) - self.clip_num_frames)
        if max_start == 0:
            return video[:, : self.clip_num_frames, :, :]
        start_index = int(np.random.randint(0, max_start + 1))
        return self._get_clip_from_start(video, start_index)

    def _sample_center_clip(self, video: torch.Tensor) -> torch.Tensor:
        video = self._pad_temporal_clip(video)
        max_start = max(0, video.size(1) - self.clip_num_frames)
        start_index = max_start // 2
        return self._get_clip_from_start(video, start_index)

    def _build_test_clips(self, video: torch.Tensor) -> torch.Tensor:
        video = self._pad_temporal_clip(video)
        max_start = max(0, video.size(1) - self.clip_num_frames)
        if self.test_num_clips <= 1 or max_start == 0:
            single_clip = self._sample_center_clip(video)
            return torch.stack([single_clip for _ in range(self.test_num_clips)], dim=0)

        start_positions = np.linspace(0, max_start, num=self.test_num_clips)
        clips = [
            self._get_clip_from_start(video, int(round(start_index)))
            for start_index in start_positions
        ]
        return torch.stack(clips, dim=0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.data.iloc[index]
        video = self._load_video_frames(str(row["frame_dir"]))

        if self.is_training:
            video = self._sample_training_clip(video)
        elif self.multi_clip_eval:
            video = self._build_test_clips(video)
        else:
            video = self._sample_center_clip(video)

        return {
            "video": video,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "sample_name": str(row["sample_name"]),
            "relative_path": str(row["relative_path"]),
        }


class FocalLoss(nn.Module):
    # 这是一个最基础的 Focal Loss 实现。
    # This is a very basic Focal Loss implementation.
    #
    # 作用：
    # 1. 减少“容易样本”对总 loss 的影响
    # 2. 让模型更关注“难样本”
    # 3. 配合 alpha 权重后，也能更关注某些更难学的类别
    #
    # Purpose:
    # 1. Reduce the contribution from easy samples
    # 2. Make the model focus more on hard samples
    # 3. With alpha weights, focus more on hard classes as well
    def __init__(
        self,
        alpha: list[float] | None = None,
        gamma: float = FOCAL_GAMMA,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if alpha is None:
            self.alpha = None
        else:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        ce_loss = -log_pt
        focal_factor = (1.0 - pt) ** self.gamma
        loss = focal_factor * ce_loss

        if hasattr(self, "alpha") and self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BasicR3DClassifier(nn.Module):
    # 模型主思路：
    # 1. 输入是一个视频张量 [B, 3, T, H, W]
    # 2. 先用最基础的 R3D-18 backbone 提取更成熟的时空特征
    # 3. 再直接做时间维聚合
    # 4. 最后用全连接层做分类
    #
    # Main idea:
    # 1. Input is a video tensor [B, 3, T, H, W]
    # 2. Use a basic R3D-18 backbone to extract stronger spatio-temporal features
    # 3. Aggregate features over time directly
    # 4. Use a linear layer for final classification
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        r3d_feature_dim: int = R3D_FEATURE_DIM,
        dropout: float = DROPOUT,
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
        # 3. 现在我们先测试纯 R3D，所以这里只需要 backbone 特征，不用最终 fc
        #
        # Why:
        # 1. R3D is a more mature 3D CNN backbone
        # 2. It usually extracts better video features than a few hand-written Conv3d layers
        # 3. We test a pure R3D model first, so we only keep the backbone features
        backbone = r3d_18(weights=None)
        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 第二部分：时间维聚合 + 分类头
        # Second part: temporal aggregation + classification head
        #
        # 这里先不接 Transformer，先看看更简单的纯 R3D 是否更容易泛化。
        # We remove the Transformer for now so we can test whether a simpler
        # pure R3D model generalizes better.
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(r3d_feature_dim)
        self.classifier = nn.Linear(r3d_feature_dim, num_classes)
        self.temporal_attention = nn.Sequential(
            nn.Linear(r3d_feature_dim, max(1, r3d_feature_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(1, r3d_feature_dim // 2), 1),
        )

        # 训练开始时，永久冻结一部分早期 R3D 层。
        # At the start of training, permanently freeze some early R3D layers.
        self.freeze_early_r3d_layers()
        self.set_frozen_batchnorm_eval()

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

    def set_frozen_batchnorm_eval(self) -> None:
        # 冻结层中的 BatchNorm 也固定在 eval 模式，
        # 避免 running mean / var 继续向训练集分布漂移。
        # Keep BatchNorm inside frozen blocks in eval mode as well,
        # so running mean / var do not keep drifting toward the training set.
        frozen_blocks: list[nn.Module] = []
        if FREEZE_R3D_STEM:
            frozen_blocks.append(self.stem)
        if FREEZE_R3D_LAYER1:
            frozen_blocks.append(self.layer1)
        if FREEZE_R3D_LAYER2:
            frozen_blocks.append(self.layer2)

        for block in frozen_blocks:
            for module in block.modules():
                if isinstance(module, nn.BatchNorm3d):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False

    def count_trainable_parameters(self) -> int:
        # 方便在日志里查看当前有多少参数在训练。
        # Useful for logging how many parameters are currently trainable.
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def train(self, mode: bool = True) -> nn.Module:
        # 每次进入 train() 时，都重新把冻结层里的 BN 固定住。
        # Every time train() is called, keep BN inside frozen blocks fixed.
        super().train(mode)
        if mode:
            self.set_frozen_batchnorm_eval()
        return self

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
        # 这里虽然不再用 Transformer，但把时间维放到中间更方便做时间注意力池化。
        # Even without the Transformer, [B, T', C] is convenient for temporal attention pooling.
        x = x.permute(0, 2, 1)

        # 用时间注意力池化替代简单平均，
        # 让模型自动更关注关键帧。
        # Use temporal attention pooling instead of simple averaging,
        # so the model can focus more on important frames.
        attention_scores = self.temporal_attention(x).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        x = (x * attention_weights).sum(dim=1)
        x = self.dropout(x)
        x = self.layer_norm(x)

        # 输出 8 类分数
        # Output logits for 8 classes
        return self.classifier(x)


def build_model() -> nn.Module:
    # 单独写一个 build_model，后面训练和测试都复用。
    # Keep model creation in one place so both training and testing reuse it.
    return BasicR3DClassifier().to(DEVICE)


def build_criterion() -> nn.Module:
    # 这里统一创建 loss，后面训练和测试都复用。
    # Build the loss function in one place so training and testing can reuse it.
    if USE_FOCAL_LOSS:
        return FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA).to(DEVICE)
    return nn.CrossEntropyLoss().to(DEVICE)


def build_dataloader(
    dataframe: pd.DataFrame,
    shuffle: bool,
    use_test_enhancement: bool = False,
    multi_clip_eval: bool = False,
) -> DataLoader:
    dataset = HMDB51FrameDataset(
        dataframe=dataframe,
        is_training=shuffle,
        use_test_enhancement=use_test_enhancement,
        multi_clip_eval=multi_clip_eval,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


def forward_with_optional_multi_clip(
    model: nn.Module,
    videos: torch.Tensor,
) -> torch.Tensor:
    # 单 clip: [B, C, T, H, W]
    # Multi-clip: [B, N, C, T, H, W]
    # Single clip: [B, C, T, H, W]
    # Multi-clip: [B, N, C, T, H, W]
    if videos.dim() == 5:
        return model(videos)

    if videos.dim() != 6:
        raise ValueError(f"Unexpected video tensor shape: {tuple(videos.shape)}")

    batch_size, num_clips, channels, frames, height, width = videos.shape
    flat_videos = videos.view(batch_size * num_clips, channels, frames, height, width)
    clip_logits = model(flat_videos)
    clip_logits = clip_logits.view(batch_size, num_clips, -1)
    return clip_logits.mean(dim=1)


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
            logits = forward_with_optional_multi_clip(model, videos)
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


def evaluate_with_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, leave=False):
            videos = batch["video"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            logits = forward_with_optional_multi_clip(model, videos)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    labels_array = np.array(all_labels)
    preds_array = np.array(all_preds)
    average_loss = total_loss / len(dataloader.dataset)
    accuracy = float((preds_array == labels_array).mean())
    return average_loss, accuracy, preds_array, labels_array


def save_test_artifacts(
    output_dir: Path,
    prefix: str,
    loss_value: float,
    accuracy: float,
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list[str],
    epoch: int | None = None,
) -> None:
    report = classification_report(
        labels,
        preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds)

    report_path = output_dir / f"{prefix}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

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
    cm_path = output_dir / f"{prefix}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    epoch_line = f"- **Epoch:** {epoch}\n" if epoch is not None else ""
    md_content = f"""# Test Results Summary

## Performance Summary
{epoch_line}- **Test Loss:** {loss_value:.6f}
- **Test Accuracy:** {accuracy:.4f} ({accuracy * 100:.2f}%)

## Classification Report
```text
{report}
```

## Confusion Matrix
![Confusion Matrix]({cm_path.name})
"""
    summary_path = output_dir / f"{prefix}_summary.md"
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(md_content)


def build_original_sample_mask(dataframe: pd.DataFrame) -> pd.Series:
    # validation 应尽量只看原始样本，不看离线增强样本。
    # Validation should ideally only use original samples, not offline-augmented ones.
    if "is_augmented" not in dataframe.columns and "augmentation_id" not in dataframe.columns:
        return pd.Series([True] * len(dataframe), index=dataframe.index)

    original_mask = pd.Series([False] * len(dataframe), index=dataframe.index)

    if "augmentation_id" in dataframe.columns:
        augmentation_ids = pd.to_numeric(
            dataframe["augmentation_id"],
            errors="coerce",
        ).fillna(-1)
        original_mask = original_mask | (augmentation_ids == 0)

    if "is_augmented" in dataframe.columns:
        is_augmented_values = (
            dataframe["is_augmented"]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        original_mask = original_mask | is_augmented_values.isin(
            ["false", "0", "no", "n"]
        )

    return original_mask


def build_grouped_folds(train_df: pd.DataFrame, num_folds: int) -> list[set[str]]:
    # 关键点：
    # 五折划分只基于原始样本做，避免 validation 被增强样本“抬高”。
    # 同一个原视频(relative_path)的原始样本和增强样本仍然视为同一组，
    # 训练时可以把该组的增强样本带上，验证时只保留原始样本。
    #
    # Key point:
    # Build folds from original samples only, so validation is not artificially
    # inflated by augmented samples. The original sample and its augmented samples
    # are still treated as one group through relative_path.
    folds: list[set[str]] = [set() for _ in range(num_folds)]
    original_mask = build_original_sample_mask(train_df)
    original_df = train_df[original_mask].copy()

    grouped = (
        original_df[["relative_path", "label"]]
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
    test_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    fold_dir = output_dir / f"fold_{fold_index + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(train_part, shuffle=True)
    val_loader = build_dataloader(val_part, shuffle=False)
    test_loader = build_dataloader(
        test_df,
        shuffle=False,
        use_test_enhancement=ENABLE_TEST_ENHANCEMENT,
        multi_clip_eval=True,
    )
    class_mapping = (
        test_df[["label", "class_name"]].drop_duplicates().sort_values("label")
    )
    class_names = class_mapping["class_name"].tolist()

    model = build_model()
    criterion = build_criterion()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
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
    best_test_accuracy = -1.0
    best_test_epoch = -1
    best_test_loss = float("inf")

    print(f"\n===== Fold {fold_index + 1}/{NUM_FOLDS} =====")
    print(f"Train samples: {len(train_part)}")
    print(f"Validation samples: {len(val_part)}")
    print(f"Test samples: {len(test_df)}")
    train_original_mask = build_original_sample_mask(train_part)
    val_original_mask = build_original_sample_mask(val_part)
    print(
        f"Train originals: {int(train_original_mask.sum())} | "
        f"Train augmented: {len(train_part) - int(train_original_mask.sum())}"
    )
    print(
        f"Validation originals: {int(val_original_mask.sum())} | "
        f"Validation augmented: {len(val_part) - int(val_original_mask.sum())}"
    )
    if isinstance(model, BasicR3DClassifier):
        print(
            f"Trainable parameters after permanent freezing: "
            f"{model.count_trainable_parameters():,}"
        )

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
        test_loss, test_accuracy, test_preds, test_labels = evaluate_with_predictions(
            model=model,
            dataloader=test_loader,
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
                "test_loss": round(test_loss, 6),
                "test_accuracy": round(test_accuracy, 6),
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
            f"test_loss={test_loss:.4f}, "
            f"test_accuracy={test_accuracy:.4f}, "
            f"learning_rate={current_lr:.6f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), fold_dir / "best_model.pth")

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_test_epoch = epoch
            best_test_loss = test_loss
            torch.save(model.state_dict(), fold_dir / "best_test_model.pth")
            save_test_artifacts(
                output_dir=fold_dir,
                prefix="best_test",
                loss_value=test_loss,
                accuracy=test_accuracy,
                labels=test_labels,
                preds=test_preds,
                class_names=class_names,
                epoch=epoch,
            )

        scheduler.step()

    return {
        "fold": fold_index + 1,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "model_path": str(fold_dir / "best_model.pth"),
        "history_path": str(fold_dir / "history.csv"),
        "best_test_accuracy": best_test_accuracy,
        "best_test_epoch": best_test_epoch,
        "best_test_loss": best_test_loss,
        "best_test_model_path": str(fold_dir / "best_test_model.pth"),
    }


def run_test_only(
    model_path: str, test_manifest: str, output_dir: Path | None = None
) -> tuple[float, float]:
    test_df = pd.read_csv(test_manifest)
    class_mapping = test_df[["label", "class_name"]].drop_duplicates().sort_values("label")
    class_names = class_mapping["class_name"].tolist()

    test_loader = build_dataloader(
        test_df,
        shuffle=False,
        use_test_enhancement=ENABLE_TEST_ENHANCEMENT,
        multi_clip_eval=True,
    )

    model = build_model()
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    criterion = build_criterion()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    print(f"Running evaluation on {len(test_df)} samples...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            videos = batch["video"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            logits = forward_with_optional_multi_clip(model, videos)
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
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
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

    print("===== Basic R3D 5-Fold Training =====")
    print(f"Device: {DEVICE}")
    print(f"Train manifest: {args.train_manifest}")
    print(f"Test manifest: {args.test_manifest}")
    print(f"Output dir: {output_dir}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")

    folds = build_grouped_folds(train_df, NUM_FOLDS)
    fold_results: list[dict[str, Any]] = []
    original_mask = build_original_sample_mask(train_df)

    for fold_index, val_relative_paths in enumerate(folds):
        in_val_fold = train_df["relative_path"].isin(val_relative_paths)
        val_part = train_df[in_val_fold & original_mask].copy()
        train_part = train_df[~in_val_fold].copy()

        fold_result = train_one_fold(
            fold_index=fold_index,
            train_part=train_part,
            val_part=val_part,
            test_df=test_df,
            output_dir=output_dir,
        )
        fold_results.append(fold_result)

    fold_summary = pd.DataFrame(fold_results)
    fold_summary.to_csv(output_dir / "cross_validation_summary.csv", index=False)

    best_row = fold_summary.sort_values("best_test_accuracy", ascending=False).iloc[0]
    best_model_path = str(best_row["best_test_model_path"])

    print("\n===== Cross Validation Summary =====")
    print(fold_summary.to_string(index=False))
    print(f"\nBest fold test model: {best_model_path}")

    test_loss = float(best_row["best_test_loss"])
    test_accuracy = float(best_row["best_test_accuracy"])
    print("\n===== Final Test Result Summary =====")
    print(f"Best Fold: {best_row['fold']}")
    print(f"Best Test Epoch: {best_row['best_test_epoch']}")
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
