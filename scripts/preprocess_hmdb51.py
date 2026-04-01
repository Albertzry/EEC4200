#!/usr/bin/env python3
"""
Preprocess HMDB51-style videos into fixed-length frame folders.

Main features:
- resize every frame to a fixed size
- resample videos to a target FPS
- generate a fixed number of frames for every sample
- create augmented versions for each training video
- export train/test manifest files for later dataloader use
"""

from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class ResizeConfig:
    width: int
    height: int


@dataclass
class LowLightEnhancementConfig:
    enabled: bool
    gamma: float
    clahe_clip_limit: float
    clahe_grid_size: int
    brightness_gain: float
    contrast_gain: float


@dataclass
class PreprocessConfig:
    dataset_name: str
    root_dir: str
    train_list: str
    test_list: str
    output_dir: str
    target_fps: float
    num_frames: int
    num_augmentations: int
    augmentation_profile: str
    random_seed: int
    jpeg_quality: int
    allowed_extensions: list[str]
    resize: ResizeConfig
    low_light_enhancement: LowLightEnhancementConfig
    augmentations: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess HMDB51-style videos into frame folders."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hmdb51_preprocess.json",
        help="Path to preprocessing config JSON.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Optional override for dataset root directory.",
    )
    parser.add_argument(
        "--train-list",
        type=str,
        default=None,
        help="Optional override for train split txt path.",
    )
    parser.add_argument(
        "--test-list",
        type=str,
        default=None,
        help="Optional override for test split txt path.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override for output directory.",
    )
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def build_runtime_config(args: argparse.Namespace) -> PreprocessConfig:
    raw_config = load_json(args.config)
    dataset_cfg = raw_config["dataset"]
    preprocess_cfg = raw_config["preprocess"]
    low_light_cfg = preprocess_cfg.get("low_light_enhancement", {})

    return PreprocessConfig(
        dataset_name=dataset_cfg.get("name", "dataset"),
        root_dir=args.dataset_root or dataset_cfg["root_dir"],
        train_list=args.train_list or dataset_cfg["train_list"],
        test_list=args.test_list or dataset_cfg["test_list"],
        output_dir=args.output_dir or preprocess_cfg["output_dir"],
        target_fps=float(preprocess_cfg["target_fps"]),
        num_frames=int(preprocess_cfg["num_frames"]),
        num_augmentations=int(preprocess_cfg["num_augmentations"]),
        augmentation_profile=str(preprocess_cfg["augmentation_profile"]),
        random_seed=int(preprocess_cfg["random_seed"]),
        jpeg_quality=int(preprocess_cfg.get("jpeg_quality", 95)),
        allowed_extensions=list(preprocess_cfg.get("allowed_extensions", [".avi"])),
        resize=ResizeConfig(
            width=int(preprocess_cfg["target_width"]),
            height=int(preprocess_cfg["target_height"]),
        ),
        low_light_enhancement=LowLightEnhancementConfig(
            enabled=bool(low_light_cfg.get("enabled", False)),
            gamma=float(low_light_cfg.get("gamma", 1.0)),
            clahe_clip_limit=float(low_light_cfg.get("clahe_clip_limit", 0.0)),
            clahe_grid_size=int(low_light_cfg.get("clahe_grid_size", 8)),
            brightness_gain=float(low_light_cfg.get("brightness_gain", 1.0)),
            contrast_gain=float(low_light_cfg.get("contrast_gain", 1.0)),
        ),
        augmentations=dict(preprocess_cfg["augmentations"]),
    )


def parse_split_file(split_path: str, split_name: str, dataset_root: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(split_path, "r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != 3:
                parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Could not parse line {line_number} in {split_path}: {raw_line!r}"
                )

            sample_id, label, relative_path = parts
            relative_path = str(Path(relative_path))
            class_name = Path(relative_path).parts[0]
            samples.append(
                {
                    "split": split_name,
                    "sample_id": int(sample_id),
                    "label": int(label),
                    "class_name": class_name,
                    "relative_path": relative_path,
                    "video_path": str(Path(dataset_root) / relative_path),
                }
            )
    return samples


def apply_low_light_enhancement(
    frame: np.ndarray,
    config: LowLightEnhancementConfig,
) -> np.ndarray:
    if not config.enabled:
        return frame

    output = frame.copy()

    # 用 gamma 提亮暗部。gamma > 1 时会让暗区域更亮。
    # Use gamma correction to brighten dark regions. gamma > 1 makes dark pixels brighter.
    if abs(config.gamma - 1.0) > 1e-6:
        gamma_value = max(config.gamma, 1e-6)
        table = np.array(
            [((index / 255.0) ** (1.0 / gamma_value)) * 255 for index in range(256)],
            dtype=np.uint8,
        )
        output = cv2.LUT(output, table)

    # CLAHE 只在亮度通道上做局部对比度增强。
    # Apply CLAHE only on the luminance channel for local contrast enhancement.
    if config.clahe_clip_limit > 0:
        grid_size = max(1, config.clahe_grid_size)
        lab = cv2.cvtColor(output, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(grid_size, grid_size),
        )
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        output = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 最后再做轻度整体提亮和对比度增强。
    # Finally apply mild global brightness and contrast enhancement.
    output_float = output.astype(np.float32)
    if abs(config.brightness_gain - 1.0) > 1e-6:
        output_float = output_float * config.brightness_gain
    if abs(config.contrast_gain - 1.0) > 1e-6:
        mean = output_float.mean(axis=(0, 1), keepdims=True)
        output_float = (output_float - mean) * config.contrast_gain + mean
    return np.clip(output_float, 0, 255).astype(np.uint8)


def read_video_frames(
    video_path: str,
    resize: ResizeConfig,
    low_light_enhancement: LowLightEnhancementConfig,
) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps is None or math.isnan(original_fps) or original_fps <= 0:
        original_fps = 30.0

    frames: list[np.ndarray] = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        enhanced = apply_low_light_enhancement(frame, low_light_enhancement)
        resized = cv2.resize(
            enhanced,
            (resize.width, resize.height),
            interpolation=cv2.INTER_LINEAR,
        )
        frames.append(resized)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    return frames, float(original_fps)


def build_resampled_indices(frame_count: int, original_fps: float, target_fps: float) -> list[int]:
    if frame_count <= 0:
        return []
    if frame_count == 1:
        return [0]

    duration_sec = frame_count / original_fps
    target_count = max(1, int(round(duration_sec * target_fps)))
    target_times = np.linspace(0.0, duration_sec, num=target_count, endpoint=False)

    indices: list[int] = []
    for time_point in target_times:
        original_index = int(round(time_point * original_fps))
        indices.append(min(frame_count - 1, max(0, original_index)))
    return indices


def resample_frames_to_target_fps(
    frames: list[np.ndarray],
    original_fps: float,
    target_fps: float,
) -> list[np.ndarray]:
    indices = build_resampled_indices(len(frames), original_fps, target_fps)
    return [frames[index] for index in indices]


def compute_frame_difference_scores(frames: list[np.ndarray]) -> list[float]:
    if len(frames) <= 1:
        return [0.0 for _ in frames]

    gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    scores = [0.0]
    for index in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[index], gray_frames[index - 1])
        scores.append(float(diff.mean()))
    return scores


def build_uniform_indices(total_count: int, wanted_count: int) -> list[int]:
    if total_count <= 0:
        return []
    if wanted_count <= 0:
        return []
    if total_count == 1:
        return [0] * wanted_count

    positions = np.linspace(0, total_count - 1, num=wanted_count)
    return [int(round(position)) for position in positions]


def pad_indices(indices: list[int], target_count: int) -> list[int]:
    if not indices:
        return []
    if len(indices) >= target_count:
        return indices[:target_count]

    padded = list(indices)
    extra_positions = np.linspace(0, len(indices) - 1, num=target_count - len(indices))
    for position in extra_positions:
        padded.append(indices[int(round(position))])
    padded.sort()
    return padded[:target_count]


def select_fixed_length_frames(frames: list[np.ndarray], target_count: int) -> list[np.ndarray]:
    frame_count = len(frames)
    if frame_count == target_count:
        return frames

    if frame_count < target_count:
        base_indices = list(range(frame_count))
        padded_indices = pad_indices(base_indices, target_count)
        return [frames[index] for index in padded_indices]

    difference_scores = compute_frame_difference_scores(frames)
    key_frame_count = min(16, target_count, frame_count)
    top_key_indices = sorted(
        np.argsort(np.array(difference_scores))[-key_frame_count:].tolist()
    )

    uniform_count = target_count - len(top_key_indices)
    uniform_indices = build_uniform_indices(frame_count, uniform_count)

    merged_indices = sorted(set(top_key_indices + uniform_indices))
    if len(merged_indices) < target_count:
        merged_indices = pad_indices(merged_indices, target_count)
    elif len(merged_indices) > target_count:
        keep_positions = build_uniform_indices(len(merged_indices), target_count)
        merged_indices = [merged_indices[index] for index in keep_positions]

    return [frames[index] for index in merged_indices]


def sanitize_sample_name(relative_path: str) -> str:
    stem = Path(relative_path).stem
    return stem.replace(" ", "_")


def sample_transform_parameters(
    augmentation_config: dict[str, Any],
    random_state: random.Random,
) -> dict[str, Any]:
    params: dict[str, Any] = {}

    flip_cfg = augmentation_config["horizontal_flip"]
    params["horizontal_flip"] = (
        flip_cfg["enabled"] and random_state.random() < flip_cfg["probability"]
    )

    brightness_cfg = augmentation_config["brightness"]
    params["brightness_factor"] = None
    if brightness_cfg["enabled"] and random_state.random() < brightness_cfg["probability"]:
        params["brightness_factor"] = random_state.uniform(
            brightness_cfg["min_factor"], brightness_cfg["max_factor"]
        )

    contrast_cfg = augmentation_config["contrast"]
    params["contrast_factor"] = None
    if contrast_cfg["enabled"] and random_state.random() < contrast_cfg["probability"]:
        params["contrast_factor"] = random_state.uniform(
            contrast_cfg["min_factor"], contrast_cfg["max_factor"]
        )

    saturation_cfg = augmentation_config["saturation"]
    params["saturation_factor"] = None
    if saturation_cfg["enabled"] and random_state.random() < saturation_cfg["probability"]:
        params["saturation_factor"] = random_state.uniform(
            saturation_cfg["min_factor"], saturation_cfg["max_factor"]
        )

    hue_cfg = augmentation_config["hue_shift"]
    params["hue_shift"] = None
    if hue_cfg["enabled"] and random_state.random() < hue_cfg["probability"]:
        params["hue_shift"] = random_state.randint(-hue_cfg["max_shift"], hue_cfg["max_shift"])

    blur_cfg = augmentation_config["gaussian_blur"]
    params["blur_kernel"] = None
    if blur_cfg["enabled"] and random_state.random() < blur_cfg["probability"]:
        params["blur_kernel"] = int(random_state.choice(blur_cfg["kernel_sizes"]))

    noise_cfg = augmentation_config["gaussian_noise"]
    params["noise_std"] = None
    if noise_cfg["enabled"] and random_state.random() < noise_cfg["probability"]:
        params["noise_std"] = random_state.uniform(noise_cfg["std_min"], noise_cfg["std_max"])

    rotation_cfg = augmentation_config["rotation"]
    params["rotation_degrees"] = 0.0
    if rotation_cfg["enabled"] and random_state.random() < rotation_cfg["probability"]:
        params["rotation_degrees"] = random_state.uniform(
            -rotation_cfg["max_degrees"], rotation_cfg["max_degrees"]
        )

    translation_cfg = augmentation_config["translation"]
    params["translate_x_ratio"] = 0.0
    params["translate_y_ratio"] = 0.0
    if translation_cfg["enabled"] and random_state.random() < translation_cfg["probability"]:
        params["translate_x_ratio"] = random_state.uniform(
            -translation_cfg["max_ratio_x"], translation_cfg["max_ratio_x"]
        )
        params["translate_y_ratio"] = random_state.uniform(
            -translation_cfg["max_ratio_y"], translation_cfg["max_ratio_y"]
        )

    zoom_cfg = augmentation_config["zoom"]
    params["zoom_factor"] = 1.0
    if zoom_cfg["enabled"] and random_state.random() < zoom_cfg["probability"]:
        params["zoom_factor"] = random_state.uniform(
            zoom_cfg["min_scale"], zoom_cfg["max_scale"]
        )

    crop_cfg = augmentation_config["random_crop"]
    params["crop_scale"] = None
    if crop_cfg["enabled"] and random_state.random() < crop_cfg["probability"]:
        params["crop_scale"] = random_state.uniform(crop_cfg["min_scale"], crop_cfg["max_scale"])
        params["crop_offset_x"] = random_state.random()
        params["crop_offset_y"] = random_state.random()
    else:
        params["crop_offset_x"] = 0.0
        params["crop_offset_y"] = 0.0

    erasing_cfg = augmentation_config["random_erasing"]
    params["erase_box"] = None
    if erasing_cfg["enabled"] and random_state.random() < erasing_cfg["probability"]:
        params["erase_area_ratio"] = random_state.uniform(
            erasing_cfg["min_area_ratio"], erasing_cfg["max_area_ratio"]
        )
        params["erase_aspect_ratio"] = random_state.uniform(
            1.0 / erasing_cfg["max_aspect_ratio"],
            erasing_cfg["max_aspect_ratio"],
        )
        params["erase_offset_x"] = random_state.random()
        params["erase_offset_y"] = random_state.random()
    else:
        params["erase_area_ratio"] = None
        params["erase_aspect_ratio"] = None
        params["erase_offset_x"] = 0.0
        params["erase_offset_y"] = 0.0

    return params


def list_applied_transforms(params: dict[str, Any]) -> list[str]:
    applied: list[str] = []
    if params["horizontal_flip"]:
        applied.append("horizontal_flip")
    if params["brightness_factor"] is not None:
        applied.append("brightness")
    if params["contrast_factor"] is not None:
        applied.append("contrast")
    if params["saturation_factor"] is not None:
        applied.append("saturation")
    if params["hue_shift"] is not None:
        applied.append("hue_shift")
    if params["blur_kernel"] is not None:
        applied.append("gaussian_blur")
    if params["noise_std"] is not None:
        applied.append("gaussian_noise")
    if abs(params["rotation_degrees"]) > 1e-6:
        applied.append("rotation")
    if abs(params["translate_x_ratio"]) > 1e-6 or abs(params["translate_y_ratio"]) > 1e-6:
        applied.append("translation")
    if abs(params["zoom_factor"] - 1.0) > 1e-6:
        applied.append("zoom")
    if params["crop_scale"] is not None:
        applied.append("random_crop")
    if params["erase_area_ratio"] is not None:
        applied.append("random_erasing")
    return applied


def build_non_empty_transform_params(
    augmentation_config: dict[str, Any],
    random_state: random.Random,
) -> tuple[dict[str, Any], list[str]]:
    for _ in range(5):
        params = sample_transform_parameters(augmentation_config, random_state)
        applied = list_applied_transforms(params)
        if applied:
            return params, applied

    params = sample_transform_parameters(augmentation_config, random_state)
    params["horizontal_flip"] = True
    return params, list_applied_transforms(params)


def apply_brightness(frame: np.ndarray, factor: float) -> np.ndarray:
    output = frame.astype(np.float32) * factor
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_contrast(frame: np.ndarray, factor: float) -> np.ndarray:
    mean = frame.mean(axis=(0, 1), keepdims=True)
    output = (frame.astype(np.float32) - mean) * factor + mean
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_saturation_and_hue(
    frame: np.ndarray,
    saturation_factor: float | None,
    hue_shift: int | None,
) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    if saturation_factor is not None:
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    if hue_shift is not None:
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_affine_transform(
    frame: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    height, width = frame.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, params["rotation_degrees"], params["zoom_factor"])
    matrix[0, 2] += params["translate_x_ratio"] * width
    matrix[1, 2] += params["translate_y_ratio"] * height
    return cv2.warpAffine(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def apply_random_crop(frame: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    crop_scale = params["crop_scale"]
    if crop_scale is None:
        return frame

    height, width = frame.shape[:2]
    crop_width = max(1, int(width * crop_scale))
    crop_height = max(1, int(height * crop_scale))
    max_x = max(0, width - crop_width)
    max_y = max(0, height - crop_height)
    start_x = int(round(max_x * params["crop_offset_x"]))
    start_y = int(round(max_y * params["crop_offset_y"]))
    cropped = frame[start_y:start_y + crop_height, start_x:start_x + crop_width]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def apply_random_erasing(frame: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    if params["erase_area_ratio"] is None or params["erase_aspect_ratio"] is None:
        return frame

    output = frame.copy()
    height, width = output.shape[:2]
    erase_area = height * width * params["erase_area_ratio"]
    erase_height = int(round(math.sqrt(erase_area / params["erase_aspect_ratio"])))
    erase_width = int(round(erase_height * params["erase_aspect_ratio"]))

    erase_width = min(max(1, erase_width), max(1, int(width * 0.35)))
    erase_height = min(max(1, erase_height), max(1, int(height * 0.35)))
    max_x = max(0, width - erase_width)
    max_y = max(0, height - erase_height)
    start_x = int(round(max_x * params["erase_offset_x"]))
    start_y = int(round(max_y * params["erase_offset_y"]))
    fill_value = output.mean(axis=(0, 1)).astype(np.uint8)
    output[start_y:start_y + erase_height, start_x:start_x + erase_width] = fill_value
    return output


def apply_gaussian_noise(frame: np.ndarray, std_value: float) -> np.ndarray:
    noise = np.random.normal(0.0, std_value, frame.shape).astype(np.float32)
    output = frame.astype(np.float32) + noise
    return np.clip(output, 0, 255).astype(np.uint8)


def apply_transforms_to_frames(
    frames: list[np.ndarray],
    params: dict[str, Any],
) -> list[np.ndarray]:
    augmented_frames: list[np.ndarray] = []
    for frame in frames:
        transformed = frame.copy()

        if params["horizontal_flip"]:
            transformed = cv2.flip(transformed, 1)

        if params["brightness_factor"] is not None:
            transformed = apply_brightness(transformed, params["brightness_factor"])

        if params["contrast_factor"] is not None:
            transformed = apply_contrast(transformed, params["contrast_factor"])

        if params["saturation_factor"] is not None or params["hue_shift"] is not None:
            transformed = apply_saturation_and_hue(
                transformed,
                params["saturation_factor"],
                params["hue_shift"],
            )

        transformed = apply_affine_transform(transformed, params)
        transformed = apply_random_crop(transformed, params)
        transformed = apply_random_erasing(transformed, params)

        if params["blur_kernel"] is not None:
            transformed = cv2.GaussianBlur(
                transformed, (params["blur_kernel"], params["blur_kernel"]), 0
            )

        if params["noise_std"] is not None:
            transformed = apply_gaussian_noise(transformed, params["noise_std"])

        augmented_frames.append(transformed)

    return augmented_frames


def save_frames_to_directory(
    frames: list[np.ndarray],
    output_dir: Path,
    jpeg_quality: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        file_path = output_dir / f"frame_{index:03d}.jpg"
        cv2.imwrite(str(file_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])


def create_manifest_record(
    sample: dict[str, Any],
    frame_dir: Path,
    sample_name: str,
    num_frames: int,
    is_augmented: bool,
    augmentation_id: int,
    applied_transforms: list[str],
) -> dict[str, Any]:
    return {
        "split": sample["split"],
        "class_name": sample["class_name"],
        "label": sample["label"],
        "sample_id": sample["sample_id"],
        "source_video": sample["video_path"],
        "relative_path": sample["relative_path"],
        "sample_name": sample_name,
        "frame_dir": str(frame_dir),
        "num_frames": num_frames,
        "is_augmented": is_augmented,
        "augmentation_id": augmentation_id,
        "applied_transforms": ",".join(applied_transforms),
    }


def process_single_video(
    sample: dict[str, Any],
    config: PreprocessConfig,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    video_path = Path(sample["video_path"])
    summary = {
        "success": False,
        "error": None,
        "generated_records": 0,
    }

    if not video_path.exists():
        summary["error"] = "missing_file"
        return [], summary

    if video_path.suffix.lower() not in config.allowed_extensions:
        summary["error"] = "unsupported_extension"
        return [], summary

    try:
        frames, original_fps = read_video_frames(
            str(video_path),
            config.resize,
            config.low_light_enhancement,
        )
        fps_frames = resample_frames_to_target_fps(frames, original_fps, config.target_fps)
        selected_frames = select_fixed_length_frames(fps_frames, config.num_frames)
    except Exception as error:  # pragma: no cover
        summary["error"] = str(error)
        return [], summary

    split_root = Path(config.output_dir) / sample["split"] / sample["class_name"]
    base_name = sanitize_sample_name(sample["relative_path"])
    records: list[dict[str, Any]] = []

    original_sample_name = f"{base_name}_orig"
    original_output_dir = split_root / original_sample_name
    save_frames_to_directory(selected_frames, original_output_dir, config.jpeg_quality)
    records.append(
        create_manifest_record(
            sample=sample,
            frame_dir=original_output_dir,
            sample_name=original_sample_name,
            num_frames=len(selected_frames),
            is_augmented=False,
            augmentation_id=0,
            applied_transforms=[],
        )
    )

    if sample["split"] == "train":
        for augmentation_index in range(1, config.num_augmentations + 1):
            params, applied_transforms = build_non_empty_transform_params(config.augmentations, rng)
            augmented_frames = apply_transforms_to_frames(selected_frames, params)
            augmented_sample_name = f"{base_name}_aug{augmentation_index}"
            augmented_output_dir = split_root / augmented_sample_name
            save_frames_to_directory(augmented_frames, augmented_output_dir, config.jpeg_quality)
            records.append(
                create_manifest_record(
                    sample=sample,
                    frame_dir=augmented_output_dir,
                    sample_name=augmented_sample_name,
                    num_frames=len(augmented_frames),
                    is_augmented=True,
                    augmentation_id=augmentation_index,
                    applied_transforms=applied_transforms,
                )
            )

    summary["success"] = True
    summary["generated_records"] = len(records)
    return records, summary


def build_summary(
    all_samples: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    config: PreprocessConfig,
) -> dict[str, Any]:
    train_original_samples = sum(1 for sample in all_samples if sample["split"] == "train")
    test_original_samples = sum(1 for sample in all_samples if sample["split"] == "test")

    summary = {
        "dataset_name": config.dataset_name,
        "target_width": config.resize.width,
        "target_height": config.resize.height,
        "target_fps": config.target_fps,
        "num_frames": config.num_frames,
        "num_augmentations": config.num_augmentations,
        "augmentation_profile": config.augmentation_profile,
        "train_original_samples": train_original_samples,
        "test_original_samples": test_original_samples,
        "train_output_samples": len(train_records),
        "test_output_samples": len(test_records),
        "failed_videos": len(failures),
        "failures": failures,
        "train_class_counts": pd.DataFrame(train_records)["class_name"].value_counts().to_dict()
        if train_records
        else {},
        "test_class_counts": pd.DataFrame(test_records)["class_name"].value_counts().to_dict()
        if test_records
        else {},
    }
    return summary


def write_outputs(
    output_dir: Path,
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_records).to_csv(output_dir / "train_manifest.csv", index=False)
    pd.DataFrame(test_records).to_csv(output_dir / "test_manifest.csv", index=False)
    with open(output_dir / "preprocess_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    config = build_runtime_config(args)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.random_seed)
    np.random.seed(config.random_seed)

    train_samples = parse_split_file(config.train_list, "train", config.root_dir)
    test_samples = parse_split_file(config.test_list, "test", config.root_dir)
    all_samples = train_samples + test_samples

    train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    num_workers = 12
    print(f"Starting multi-threaded preprocessing with {num_workers} workers...")

    # 为了保证多线程下的随机性，我们为每个任务传入一个独立的种子偏移
    # To ensure randomness in multi-threading, we pass a unique seed offset for each task
    def task_wrapper(sample):
        # 使用原始随机种子加上 sample_id 确保每个视频的增强是确定且唯一的
        # Use base seed + sample_id to ensure augmentations are deterministic and unique
        task_rng = random.Random(config.random_seed + sample["sample_id"])
        records, item_summary = process_single_video(sample, config, task_rng)
        return records, item_summary, sample

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(task_wrapper, sample): sample for sample in all_samples}

        for future in tqdm(as_completed(futures), total=len(all_samples), desc="Preprocessing videos"):
            records, item_summary, sample = future.result()
            if item_summary["success"]:
                if sample["split"] == "train":
                    train_records.extend(records)
                else:
                    test_records.extend(records)
            else:
                failures.append(
                    {
                        "split": sample["split"],
                        "class_name": sample["class_name"],
                        "relative_path": sample["relative_path"],
                        "error": item_summary["error"],
                    }
                )

    summary = build_summary(
        all_samples=all_samples,
        train_records=train_records,
        test_records=test_records,
        failures=failures,
        config=config,
    )
    write_outputs(output_dir, train_records, test_records, summary)

    print("\n===== Preprocess Summary =====")
    print(f"Train original samples: {summary['train_original_samples']}")
    print(f"Test original samples: {summary['test_original_samples']}")
    print(f"Train output samples: {summary['train_output_samples']}")
    print(f"Test output samples: {summary['test_output_samples']}")
    print(f"Failed videos: {summary['failed_videos']}")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
