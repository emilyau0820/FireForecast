"""
preprocess.py
─────────────
Converts raw TFRecord files from the Next Day Wildfire Spread Kaggle dataset
into numpy .npy files for fast loading in PyTorch.

Download the dataset first:
  kaggle datasets download -d fantineh/next-day-wildfire-spread -p data/raw --unzip

Run this script once before training:
  python preprocess.py
"""

import os
import numpy as np
import glob
import yaml

# TensorFlow is only used here to read TFRecord files — not used anywhere else
import tensorflow as tf

with open("configs/base.yaml", "r") as f:
    cfg = yaml.safe_load(f)

RAW_DIR         = cfg["data"]["raw_dir"]
PROC_DIR        = cfg["data"]["processed_dir"]
INPUT_FEATURES  = cfg["features"]["input_features"]
LABEL_FEATURE   = cfg["features"]["label_feature"]
ALL_FEATURES    = INPUT_FEATURES + [LABEL_FEATURE]
IMG_SIZE        = 64


def parse_tfrecord(example_proto):
    """
    Parses a single TFRecord example.
    Each feature is a flat float32 list of length 64*64=4096.
    Reshapes each into a 64x64 spatial grid.
    Returns inputs as [12, 64, 64] and label as [1, 64, 64] numpy arrays.
    """
    feature_spec = {
        name: tf.io.FixedLenFeature([IMG_SIZE * IMG_SIZE], tf.float32)
        for name in ALL_FEATURES
    }
    parsed = tf.io.parse_single_example(example_proto, feature_spec)

    # Stack all input channels → [12, 64, 64]
    inputs = tf.stack(
        [tf.reshape(parsed[f], [IMG_SIZE, IMG_SIZE]) for f in INPUT_FEATURES],
        axis=0
    )
    # Label → [1, 64, 64]
    label = tf.reshape(parsed[LABEL_FEATURE], [1, IMG_SIZE, IMG_SIZE])

    return inputs.numpy(), label.numpy()


def convert_split(split_name, file_pattern):
    """
    Reads all TFRecord files matching file_pattern, parses every record,
    cleans NaN/inf values, and saves clean numpy arrays to the processed directory.

    Cleaning is done here at save time so that the Dataset class never has to
    handle NaN/inf, avoiding large temporary memory allocations during training.
    """
    files = sorted(glob.glob(os.path.join(RAW_DIR, file_pattern)))
    print(f"[{split_name}] Found {len(files)} TFRecord file(s)")

    all_inputs, all_labels = [], []

    for fpath in files:
        dataset = tf.data.TFRecordDataset(fpath, compression_type="")
        for raw_record in dataset:
            try:
                x, y = parse_tfrecord(raw_record)
                all_inputs.append(x)
                all_labels.append(y)
            except Exception as e:
                print(f"  Skipping malformed record: {e}")

    all_inputs = np.array(all_inputs, dtype=np.float32)  # [N, 12, 64, 64]
    all_labels = np.array(all_labels, dtype=np.float32)  # [N,  1, 64, 64]

    # ── Clean NaN and inf values channel-by-channel to keep memory usage low ──
    # Processing one channel at a time (~58 MB each) rather than the full array
    # (~700 MB) avoids the ArrayMemoryError caused by numpy's internal bool mask.
    print(f"  Cleaning NaN/inf values...")
    for c in range(all_inputs.shape[1]):
        all_inputs[:, c, :, :] = np.nan_to_num(
            all_inputs[:, c, :, :], nan=0.0, posinf=0.0, neginf=0.0
        )
    # Clean labels too (uncertain pixels are -1, which is valid — don't zero them)
    # Only replace actual NaN/inf, not the -1 uncertainty markers
    all_labels = np.nan_to_num(all_labels, nan=0.0, posinf=0.0, neginf=0.0)

    os.makedirs(PROC_DIR, exist_ok=True)
    np.save(os.path.join(PROC_DIR, f"{split_name}_x.npy"), all_inputs)
    np.save(os.path.join(PROC_DIR, f"{split_name}_y.npy"), all_labels)
    print(f"  Saved {len(all_inputs)} samples → {PROC_DIR}/{split_name}_x.npy")


def extract_metadata(tfrecord_path: str) -> None:
    """
    Inspects the first TFRecord example in tfrecord_path and prints:
      1. All feature keys present in the record.
      2. Keys whose names contain date/time/location keywords, with sample values.

    Does NOT modify any data — purely a discovery tool to check what geographic
    or temporal metadata the dataset carries alongside the 12 input channels.

    Usage:
        python -c "from preprocess import extract_metadata; \
                   extract_metadata('data/raw/next_day_wildfire_spread_train_00.tfrecord')"
    """
    METADATA_KEYWORDS = {"date", "time", "year", "month", "day", "lon", "lat", "location"}

    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        all_keys = sorted(example.features.feature.keys())

        print(f"\nAll feature keys in first record of '{tfrecord_path}'")
        print(f"  Total: {len(all_keys)}")
        for k in all_keys:
            print(f"  {k}")

        matching = [k for k in all_keys if any(kw in k.lower() for kw in METADATA_KEYWORDS)]
        print(f"\nKeys matching metadata keywords {METADATA_KEYWORDS}:")
        if not matching:
            print("  (none found)")
        for k in matching:
            feat = example.features.feature[k]
            kind = feat.WhichOneof("kind")
            if kind == "int64_list":
                vals = list(feat.int64_list.value)[:8]
                print(f"  [{kind}] {k}: {vals}")
            elif kind == "bytes_list":
                vals = [v.decode("utf-8", errors="replace") for v in feat.bytes_list.value[:8]]
                print(f"  [{kind}] {k}: {vals}")
            elif kind == "float_list":
                vals = list(feat.float_list.value)[:8]
                print(f"  [{kind}] {k}: {vals}")


if __name__ == "__main__":
    convert_split("train", cfg["data"]["train_file"])
    convert_split("val",   cfg["data"]["val_file"])
    convert_split("test",  cfg["data"]["test_file"])
    print("Preprocessing complete.")

    # Uncomment to inspect metadata keys after preprocessing:
    # extract_metadata("data/raw/next_day_wildfire_spread_train_00.tfrecord")