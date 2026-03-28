# HMDB51 Dataset Analysis Starter

This repository currently contains a beginner-friendly dataset analysis script for an HMDB51-style video classification task.

It now also contains a preprocessing script that converts videos into fixed-length frame folders for later 3D CNN / Transformer training.

It also contains a very basic PyTorch training script for a 3D CNN + Transformer classifier.

## What this script does

The script reads your train/test split files and analyzes each video to help you understand dataset quality before training a 3D CNN or a 3D CNN + Transformer model.

It checks:

- class distribution
- train/test split balance
- video duration
- frame count
- FPS
- resolution
- file size
- missing files
- unreadable videos
- duplicate relative paths across splits
- suspicious clips such as very short, very low-FPS, or very low-resolution samples

## Project structure

```text
EEC4200/
├── configs/
│   └── hmdb51.json
│   └── hmdb51_preprocess.json
├── scripts/
│   └── analyze_hmdb51.py
│   └── preprocess_hmdb51.py
│   └── train_basic_3dcnn_transformer.py
├── requirements.txt
└── README.md
```

## Path configuration

All dataset paths are stored in [configs/hmdb51.json](/Users/albert/PycharmProjects/EEC4200/configs/hmdb51.json).

You can keep one config for local development, then either:

1. edit the JSON file directly on the server, or
2. keep the JSON file the same and override paths from the command line

The current local paths are already filled in for you.

You can also override paths with environment variables:

```bash
export HMDB51_ROOT_DIR=/path/to/HMDB51
export HMDB51_TRAIN_LIST=/path/to/hmdb51_train.txt
export HMDB51_TEST_LIST=/path/to/hmdb51_test.txt
export HMDB51_OUTPUT_DIR=outputs/hmdb51_analysis_server
```

## How to run on your server

Install dependencies:

```bash
pip install -r requirements.txt
```

Run with the default config:

```bash
python scripts/analyze_hmdb51.py
```

Run preprocessing with the default config:

```bash
python scripts/preprocess_hmdb51.py
```

Run the basic training script:

```bash
python scripts/train_basic_3dcnn_transformer.py
```

Run with overridden paths:

```bash
python scripts/analyze_hmdb51.py \
  --dataset-root /path/to/HMDB51 \
  --train-list /path/to/hmdb51_train.txt \
  --test-list /path/to/hmdb51_test.txt \
  --output-dir outputs/hmdb51_analysis_server
```

You can also override preprocessing paths:

```bash
python scripts/preprocess_hmdb51.py \
  --dataset-root /path/to/HMDB51 \
  --train-list /path/to/hmdb51_train.txt \
  --test-list /path/to/hmdb51_test.txt \
  --output-dir outputs/hmdb51_preprocessed_server
```

## Preprocessing behavior

The preprocessing script does the following:

- resize all frames to `224x224`
- resample videos to `16 FPS`
- generate exactly `32` frames per sample
- use a mixed strategy for long videos:
  - keep `16` high-difference frames
  - fill the rest with `16` uniformly sampled frames
- pad short videos by repeating frames uniformly
- create `7` augmented versions for each training video
- keep the test split unaugmented

The output directory contains:

- `train/<class_name>/<sample_name>/frame_000.jpg ...`
- `test/<class_name>/<sample_name>/frame_000.jpg ...`
- `train_manifest.csv`
- `test_manifest.csv`
- `preprocess_summary.json`

## Basic training behavior

The training script:

- reads `train_manifest.csv` and `test_manifest.csv`
- loads each sample as `32` frames with shape `224x224`
- builds a very basic `3D CNN + Transformer` classifier
- trains with `CrossEntropyLoss`, `Adam`, and a simple learning-rate decay
- evaluates on the test set every epoch
- updates `training_curves.png` after every epoch
- saves:
  - `best_model.pth`
  - `history.csv`
  - `training_curves.png`

Default manifest paths:

- `outputs/hmdb51_preprocessed/train_manifest.csv`
- `outputs/hmdb51_preprocessed/test_manifest.csv`

You can override them:

```bash
python scripts/train_basic_3dcnn_transformer.py \
  --train-manifest outputs/hmdb51_preprocessed/train_manifest.csv \
  --test-manifest outputs/hmdb51_preprocessed/test_manifest.csv \
  --output-dir outputs/hmdb51_training_basic
```

## Output files

The script writes several files into the output directory:

- `video_metadata.csv`: one row per video
- `class_summary.csv`: one row per action class
- `split_summary.csv`: summary for train/test
- `issues.csv`: videos with possible problems
- `summary.json`: overall summary
- `analysis_report.md`: a beginner-friendly text report

## How to interpret the results as a beginner

- If some classes have far more videos than others, your model may become biased toward those classes.
- If some videos are much shorter or longer than most others, sampling fixed-length clips during training will need extra care.
- If FPS values vary a lot, the motion speed seen by the model may become inconsistent.
- If resolution varies a lot, you will probably resize frames during preprocessing.
- If unreadable or missing files exist, fix those before training.

## Why this matters for 3D CNN + Transformer

Your future model will likely sample a fixed number of frames from each video clip. So these statistics help you decide:

- how many frames to sample per clip
- whether to use uniform sampling
- whether short videos need padding or looping
- what input image size is reasonable
- whether your dataset is balanced enough or needs weighted loss / weighted sampling

## Suggested next step

After you run this analysis on the server, we can continue by building:

1. a training config system
2. a dataset loader for HMDB51
3. a baseline 3D CNN model
4. a 3D CNN + Transformer hybrid model
5. a train / validate / test pipeline
