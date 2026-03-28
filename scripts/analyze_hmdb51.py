#!/usr/bin/env python3
"""
Analyze HMDB51-style video datasets and export quality reports.

Expected split file format:
index <tab> label <tab> relative/path/to/video.avi

Example:
0    0    drink/clip_001.avi
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "OpenCV is required. Please install dependencies first, for example:\n"
        "pip install -r requirements.txt"
    ) from exc


@dataclass
class Thresholds:
    short_duration_sec: float
    long_duration_sec: float
    low_resolution_width: int
    low_resolution_height: int
    low_fps: float
    high_fps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze dataset quality for HMDB51-style video classification."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hmdb51.json",
        help="Path to a JSON config file.",
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
        help="Optional override for analysis output directory.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    dataset_cfg = config["dataset"]
    analysis_cfg = config["analysis"]

    env_root_dir = os.environ.get("HMDB51_ROOT_DIR")
    env_train_list = os.environ.get("HMDB51_TRAIN_LIST")
    env_test_list = os.environ.get("HMDB51_TEST_LIST")
    env_output_dir = os.environ.get("HMDB51_OUTPUT_DIR")

    runtime = {
        "dataset_name": dataset_cfg.get("name", "dataset"),
        "root_dir": args.dataset_root or env_root_dir or dataset_cfg["root_dir"],
        "train_list": args.train_list or env_train_list or dataset_cfg["train_list"],
        "test_list": args.test_list or env_test_list or dataset_cfg["test_list"],
        "output_dir": args.output_dir or env_output_dir or analysis_cfg["output_dir"],
        "num_workers": int(analysis_cfg.get("num_workers", 4)),
        "thresholds": Thresholds(**analysis_cfg["thresholds"]),
    }
    return runtime


def parse_split_file(split_path: str, split_name: str, dataset_root: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
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
            class_name = Path(relative_path).parts[0] if Path(relative_path).parts else "unknown"
            full_path = str(Path(dataset_root) / relative_path)

            samples.append(
                {
                    "split": split_name,
                    "sample_id": int(sample_id),
                    "label": int(label),
                    "class_name": class_name,
                    "relative_path": relative_path,
                    "video_path": full_path,
                }
            )
    return samples


def safe_float(value: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def inspect_video(sample: dict[str, Any], thresholds: Thresholds) -> dict[str, Any]:
    video_path = Path(sample["video_path"])
    result = dict(sample)
    result.update(
        {
            "exists": video_path.exists(),
            "readable": False,
            "file_size_mb": None,
            "frame_count": None,
            "fps": None,
            "duration_sec": None,
            "width": None,
            "height": None,
            "aspect_ratio": None,
            "resolution": None,
            "issue_missing_file": False,
            "issue_unreadable_video": False,
            "issue_short_clip": False,
            "issue_long_clip": False,
            "issue_low_resolution": False,
            "issue_low_fps": False,
            "issue_high_fps": False,
            "issue_duplicate_path": False,
        }
    )

    if not result["exists"]:
        result["issue_missing_file"] = True
        return result

    result["file_size_mb"] = safe_float(video_path.stat().st_size / (1024 * 1024))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result["issue_unreadable_video"] = True
        cap.release()
        return result

    frame_count = safe_float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = safe_float(cap.get(cv2.CAP_PROP_FPS))
    width = safe_float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = safe_float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    readable = True
    duration_sec = None
    if fps and fps > 0 and frame_count is not None:
        duration_sec = frame_count / fps
    else:
        readable = False

    result.update(
        {
            "readable": readable,
            "frame_count": int(frame_count) if frame_count is not None else None,
            "fps": round(fps, 4) if fps is not None else None,
            "duration_sec": round(duration_sec, 4) if duration_sec is not None else None,
            "width": int(width) if width is not None else None,
            "height": int(height) if height is not None else None,
            "aspect_ratio": round(width / height, 4) if width and height else None,
            "resolution": f"{int(width)}x{int(height)}" if width and height else None,
        }
    )

    if not readable:
        result["issue_unreadable_video"] = True
        return result

    result["issue_short_clip"] = duration_sec < thresholds.short_duration_sec
    result["issue_long_clip"] = duration_sec > thresholds.long_duration_sec
    result["issue_low_resolution"] = (
        (width is not None and width < thresholds.low_resolution_width)
        or (height is not None and height < thresholds.low_resolution_height)
    )
    result["issue_low_fps"] = fps < thresholds.low_fps if fps is not None else False
    result["issue_high_fps"] = fps > thresholds.high_fps if fps is not None else False

    return result


def summarize_numeric(series: pd.Series) -> dict[str, float | None]:
    clean = series.dropna()
    if clean.empty:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
        }

    return {
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "std": round(float(clean.std(ddof=0)), 4),
    }


def infer_duration_bucket(duration_sec: float | None) -> str | None:
    if duration_sec is None:
        return None
    if duration_sec < 1:
        return "<1s"
    if duration_sec < 2:
        return "1-2s"
    if duration_sec < 4:
        return "2-4s"
    if duration_sec < 8:
        return "4-8s"
    if duration_sec < 16:
        return "8-16s"
    return ">=16s"


def build_global_summary(df: pd.DataFrame) -> dict[str, Any]:
    readable_df = df[df["readable"]]

    split_counts = (
        df.groupby("split")
        .size()
        .sort_index()
        .to_dict()
    )

    class_counts = (
        df.groupby("class_name")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    imbalance_ratio = None
    if class_counts:
        counts = list(class_counts.values())
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = round(max_count / min_count, 4) if min_count > 0 else None

    resolution_counts = (
        readable_df["resolution"]
        .value_counts()
        .head(10)
        .to_dict()
    )
    fps_counts = (
        readable_df["fps"]
        .round(2)
        .value_counts()
        .head(10)
        .sort_index()
        .to_dict()
    )
    duration_bucket_counts = (
        readable_df["duration_bucket"]
        .value_counts()
        .to_dict()
    )

    summary = {
        "total_videos": int(len(df)),
        "total_classes": int(df["class_name"].nunique()),
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "class_balance": {
            "smallest_class_size": int(min(class_counts.values())) if class_counts else None,
            "largest_class_size": int(max(class_counts.values())) if class_counts else None,
            "imbalance_ratio_max_over_min": imbalance_ratio,
        },
        "availability": {
            "missing_files": int(df["issue_missing_file"].sum()),
            "unreadable_videos": int(df["issue_unreadable_video"].sum()),
            "readable_videos": int(df["readable"].sum()),
            "duplicate_relative_paths": int(df["issue_duplicate_path"].sum()),
        },
        "duration_sec": summarize_numeric(readable_df["duration_sec"]),
        "frame_count": summarize_numeric(readable_df["frame_count"]),
        "fps": summarize_numeric(readable_df["fps"]),
        "file_size_mb": summarize_numeric(readable_df["file_size_mb"]),
        "width": summarize_numeric(readable_df["width"]),
        "height": summarize_numeric(readable_df["height"]),
        "top_resolutions": resolution_counts,
        "top_fps_values": fps_counts,
        "duration_buckets": duration_bucket_counts,
        "quality_flags": {
            "short_clips": int(df["issue_short_clip"].sum()),
            "long_clips": int(df["issue_long_clip"].sum()),
            "low_resolution": int(df["issue_low_resolution"].sum()),
            "low_fps": int(df["issue_low_fps"].sum()),
            "high_fps": int(df["issue_high_fps"].sum()),
            "duplicate_paths": int(df["issue_duplicate_path"].sum()),
        },
    }
    return summary


def build_class_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for class_name, class_df in df.groupby("class_name"):
        readable_df = class_df[class_df["readable"]]
        split_counts = class_df.groupby("split").size().to_dict()
        train_videos = int(split_counts.get("train", 0))
        test_videos = int(split_counts.get("test", 0))
        rows.append(
            {
                "class_name": class_name,
                "total_videos": int(len(class_df)),
                "train_videos": train_videos,
                "test_videos": test_videos,
                "train_test_gap": train_videos - test_videos,
                "train_over_test_ratio": round(train_videos / test_videos, 4)
                if test_videos > 0
                else None,
                "missing_files": int(class_df["issue_missing_file"].sum()),
                "unreadable_videos": int(class_df["issue_unreadable_video"].sum()),
                "avg_duration_sec": round(float(readable_df["duration_sec"].mean()), 4)
                if not readable_df.empty
                else None,
                "median_duration_sec": round(float(readable_df["duration_sec"].median()), 4)
                if not readable_df.empty
                else None,
                "avg_fps": round(float(readable_df["fps"].mean()), 4)
                if not readable_df.empty
                else None,
                "median_frame_count": int(readable_df["frame_count"].median())
                if not readable_df.empty and pd.notna(readable_df["frame_count"].median())
                else None,
                "most_common_resolution": readable_df["resolution"].mode().iat[0]
                if not readable_df.empty and not readable_df["resolution"].mode().empty
                else None,
                "short_clips": int(class_df["issue_short_clip"].sum()),
                "long_clips": int(class_df["issue_long_clip"].sum()),
                "low_resolution": int(class_df["issue_low_resolution"].sum()),
                "low_fps": int(class_df["issue_low_fps"].sum()),
                "high_fps": int(class_df["issue_high_fps"].sum()),
                "duplicate_paths": int(class_df["issue_duplicate_path"].sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["total_videos", "class_name"], ascending=[False, True]
    )


def build_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, split_df in df.groupby("split"):
        readable_df = split_df[split_df["readable"]]
        class_counts = split_df["class_name"].value_counts()
        rows.append(
            {
                "split": split_name,
                "total_videos": int(len(split_df)),
                "total_classes": int(split_df["class_name"].nunique()),
                "readable_videos": int(split_df["readable"].sum()),
                "missing_files": int(split_df["issue_missing_file"].sum()),
                "unreadable_videos": int(split_df["issue_unreadable_video"].sum()),
                "avg_duration_sec": round(float(readable_df["duration_sec"].mean()), 4)
                if not readable_df.empty
                else None,
                "avg_fps": round(float(readable_df["fps"].mean()), 4)
                if not readable_df.empty
                else None,
                "avg_frame_count": round(float(readable_df["frame_count"].mean()), 4)
                if not readable_df.empty
                else None,
                "smallest_class_size": int(class_counts.min()) if not class_counts.empty else None,
                "largest_class_size": int(class_counts.max()) if not class_counts.empty else None,
            }
        )
    return pd.DataFrame(rows).sort_values("split")


def collect_issue_table(df: pd.DataFrame) -> pd.DataFrame:
    issue_mask = (
        df["issue_missing_file"]
        | df["issue_unreadable_video"]
        | df["issue_short_clip"]
        | df["issue_long_clip"]
        | df["issue_low_resolution"]
        | df["issue_low_fps"]
        | df["issue_high_fps"]
        | df["issue_duplicate_path"]
    )
    issue_columns = [
        "split",
        "class_name",
        "relative_path",
        "video_path",
        "exists",
        "readable",
        "frame_count",
        "fps",
        "duration_sec",
        "width",
        "height",
        "issue_missing_file",
        "issue_unreadable_video",
        "issue_short_clip",
        "issue_long_clip",
        "issue_low_resolution",
        "issue_low_fps",
        "issue_high_fps",
        "issue_duplicate_path",
    ]
    return df.loc[issue_mask, issue_columns].sort_values(
        by=["issue_missing_file", "issue_unreadable_video", "duration_sec"],
        ascending=[False, False, True],
        na_position="last",
    )


def build_recommendations(
    summary: dict[str, Any],
    class_summary_df: pd.DataFrame,
    split_summary_df: pd.DataFrame,
) -> list[str]:
    recommendations: list[str] = []

    imbalance_ratio = summary["class_balance"]["imbalance_ratio_max_over_min"]
    if imbalance_ratio and imbalance_ratio > 1.5:
        recommendations.append(
            "类别样本数差异明显，训练时建议考虑 class weight 或 weighted sampler。"
        )
    else:
        recommendations.append("类别分布整体较均衡，初始版本可以先不做复杂的重采样。")

    duration_stats = summary["duration_sec"]
    if duration_stats["median"] is not None:
        recommendations.append(
            f"视频时长中位数约为 {duration_stats['median']} 秒，可以据此决定每段 clip 采样多少帧。"
        )

    if summary["quality_flags"]["short_clips"] > 0:
        recommendations.append("存在很短的视频，后续数据加载器需要支持 padding、重复采样或循环取帧。")

    if summary["quality_flags"]["low_fps"] > 0 or summary["quality_flags"]["high_fps"] > 0:
        recommendations.append("不同视频的 FPS 可能不一致，建议训练时按时间均匀采样，而不是只按帧编号采样。")

    if summary["quality_flags"]["low_resolution"] > 0:
        recommendations.append("分辨率不统一，建议在预处理阶段统一 resize 到固定尺寸，例如 112x112 或 224x224。")

    if summary["availability"]["missing_files"] > 0 or summary["availability"]["unreadable_videos"] > 0:
        recommendations.append("训练前应优先清理缺失或损坏的视频，否则 dataloader 训练过程中容易中断。")

    if summary["quality_flags"]["duplicate_paths"] > 0:
        recommendations.append("发现重复路径样本，建议确认训练集和测试集之间是否存在数据泄漏。")

    if not split_summary_df.empty:
        recommendations.append("建议先用分析结果确定一个简单基线配置，再开始 3D CNN + Transformer，避免模型过早复杂化。")

    return recommendations


def write_markdown_report(
    summary: dict[str, Any],
    class_summary_df: pd.DataFrame,
    split_summary_df: pd.DataFrame,
    issue_df: pd.DataFrame,
    output_dir: str,
) -> None:
    output_path = Path(output_dir)
    report_path = output_path / "analysis_report.md"

    top_classes = class_summary_df.head(10)[
        ["class_name", "total_videos", "train_videos", "test_videos", "avg_duration_sec"]
    ]
    recommendations = build_recommendations(summary, class_summary_df, split_summary_df)

    lines = [
        "# HMDB51 Analysis Report",
        "",
        "## Overall",
        f"- Total videos: {summary['total_videos']}",
        f"- Total classes: {summary['total_classes']}",
        f"- Split counts: {summary['split_counts']}",
        f"- Readable videos: {summary['availability']['readable_videos']}",
        f"- Missing files: {summary['availability']['missing_files']}",
        f"- Unreadable videos: {summary['availability']['unreadable_videos']}",
        f"- Duplicate relative paths: {summary['availability']['duplicate_relative_paths']}",
        "",
        "## Distribution",
        f"- Class imbalance ratio (max/min): {summary['class_balance']['imbalance_ratio_max_over_min']}",
        f"- Duration buckets: {summary['duration_buckets']}",
        f"- Top resolutions: {summary['top_resolutions']}",
        f"- Top FPS values: {summary['top_fps_values']}",
        "",
        "## Numeric Summary",
        f"- Duration (sec): {summary['duration_sec']}",
        f"- Frame count: {summary['frame_count']}",
        f"- FPS: {summary['fps']}",
        f"- File size (MB): {summary['file_size_mb']}",
        f"- Width: {summary['width']}",
        f"- Height: {summary['height']}",
        "",
        "## Quality Flags",
        f"- Short clips: {summary['quality_flags']['short_clips']}",
        f"- Long clips: {summary['quality_flags']['long_clips']}",
        f"- Low resolution: {summary['quality_flags']['low_resolution']}",
        f"- Low FPS: {summary['quality_flags']['low_fps']}",
        f"- High FPS: {summary['quality_flags']['high_fps']}",
        "",
        "## Beginner Notes",
    ]

    for item in recommendations:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Top Classes By Sample Count",
            "```text",
            top_classes.to_string(index=False) if not top_classes.empty else "No class data.",
            "```",
            "",
            "## Files",
            "- `video_metadata.csv`: per-video metadata",
            "- `class_summary.csv`: per-class summary",
            "- `split_summary.csv`: train/test summary",
            "- `issues.csv`: suspicious or problematic samples",
            "- `summary.json`: machine-readable overall summary",
        ]
    )

    if not issue_df.empty:
        lines.extend(
            [
                "",
                f"## Issues",
                f"- Total flagged videos: {len(issue_df)}",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_reports(
    df: pd.DataFrame,
    class_summary_df: pd.DataFrame,
    split_summary_df: pd.DataFrame,
    issue_df: pd.DataFrame,
    global_summary: dict[str, Any],
    output_dir: str,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path / "video_metadata.csv", index=False)
    class_summary_df.to_csv(output_path / "class_summary.csv", index=False)
    split_summary_df.to_csv(output_path / "split_summary.csv", index=False)
    issue_df.to_csv(output_path / "issues.csv", index=False)

    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2, ensure_ascii=False)


def print_console_summary(summary: dict[str, Any], class_summary_df: pd.DataFrame) -> None:
    print("\n===== Dataset Quality Summary =====")
    print(f"Total videos: {summary['total_videos']}")
    print(f"Total classes: {summary['total_classes']}")
    print(f"Split counts: {summary['split_counts']}")
    print(
        "Class balance (max/min): "
        f"{summary['class_balance']['imbalance_ratio_max_over_min']}"
    )
    print(f"Missing files: {summary['availability']['missing_files']}")
    print(f"Unreadable videos: {summary['availability']['unreadable_videos']}")
    print(f"Duplicate relative paths: {summary['availability']['duplicate_relative_paths']}")
    print(f"Readable videos: {summary['availability']['readable_videos']}")
    print(f"Duration stats (sec): {summary['duration_sec']}")
    print(f"FPS stats: {summary['fps']}")
    print(f"Top resolutions: {summary['top_resolutions']}")
    print(f"Top fps values: {summary['top_fps_values']}")
    print(f"Duration buckets: {summary['duration_buckets']}")
    print(f"Quality flags: {summary['quality_flags']}")

    print("\nClasses with the most samples:")
    if class_summary_df.empty:
        print("No class summary available.")
    else:
        preview = class_summary_df.head(10)[
            ["class_name", "total_videos", "train_videos", "test_videos", "avg_duration_sec"]
        ]
        print(preview.to_string(index=False))


def main() -> None:
    args = parse_args()
    runtime = build_runtime_config(args)
    thresholds: Thresholds = runtime["thresholds"]

    train_samples = parse_split_file(
        split_path=runtime["train_list"],
        split_name="train",
        dataset_root=runtime["root_dir"],
    )
    test_samples = parse_split_file(
        split_path=runtime["test_list"],
        split_name="test",
        dataset_root=runtime["root_dir"],
    )
    samples = train_samples + test_samples

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=runtime["num_workers"]) as executor:
        futures = [executor.submit(inspect_video, sample, thresholds) for sample in samples]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Inspecting videos",
        ):
            results.append(future.result())

    df = pd.DataFrame(results).sort_values(by=["split", "class_name", "sample_id"])
    df["duration_bucket"] = df["duration_sec"].apply(infer_duration_bucket)
    df["issue_duplicate_path"] = df.duplicated(subset=["relative_path"], keep=False)
    class_summary_df = build_class_summary(df)
    split_summary_df = build_split_summary(df)
    issue_df = collect_issue_table(df)
    global_summary = build_global_summary(df)

    export_reports(
        df=df,
        class_summary_df=class_summary_df,
        split_summary_df=split_summary_df,
        issue_df=issue_df,
        global_summary=global_summary,
        output_dir=runtime["output_dir"],
    )
    write_markdown_report(
        summary=global_summary,
        class_summary_df=class_summary_df,
        split_summary_df=split_summary_df,
        issue_df=issue_df,
        output_dir=runtime["output_dir"],
    )
    print_console_summary(global_summary, class_summary_df)
    print(f"\nAnalysis files saved to: {runtime['output_dir']}")


if __name__ == "__main__":
    main()
