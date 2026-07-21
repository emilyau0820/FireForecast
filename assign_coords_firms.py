"""
assign_coords_firms.py
----------------------
Assigns APPROXIMATE geographic coordinates to each test sample by matching
against the NASA FIRMS MODIS C6.1 active fire archive.

IMPORTANT — coordinate accuracy
--------------------------------
The Kaggle "Next Day Wildfire Spread" TFRecords contain NO geographic
metadata.  This script distributes the 1,689 test-sample dots across real
western-US fire locations from the dataset's 2012-2018 period, improving the
Sample Map from "all dots at one point" to "dots spread across actual fire
zones."  Individual placements are NOT verified against the precise patch
location — treat coordinates as plausible fire-event locations, not exact
patch centroids.

Algorithm
---------
1.  Load FIRMS MODIS C6.1 CSV files for the contiguous US, 2012-2018.
2.  Filter to western US (lon < -100 degrees) where the dataset's events occur.
3.  Cluster detections spatially into discrete fire events using DBSCAN with
    haversine distance (~55 km eps).
4.  For each test sample, compute the fraction of active PrevFireMask pixels
    (channel 11) as a proxy for relative fire event size.
5.  Assign each sample to a randomly chosen event whose detection count is
    proportional to the sample's fire fraction (top-10% size-matched pool).
6.  Save data/processed/sample_metadata.csv with columns:
        sample_idx, lat_center, lon_center, date

Downloading FIRMS data (free, no account required for archive files)
--------------------------------------------------------------------
Annual archive files can be downloaded directly from:
    https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/

Download one file per year covering the US (2012-2018):
    MODIS_C6_1_USA_contiguous_and_Hawaii_<YEAR>.csv

Place all downloaded CSV files in data/raw/firms/ then run:
    python assign_coords_firms.py

Usage
-----
    python assign_coords_firms.py
    python assign_coords_firms.py --firms_dir data/raw/firms --seed 42
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# ── Constants ─────────────────────────────────────────────────────────────────

PREV_FIRE_CH   = 11       # PrevFireMask channel index in test_x.npy
WESTERN_US_LON = -100.0   # dataset events occur west of this longitude
US_LAT_MIN     = 24.0
US_LAT_MAX     = 50.0
DATASET_YR_MIN = 2012     # Huot et al. 2022 dataset spans 2012-2018
DATASET_YR_MAX = 2018

# DBSCAN: ~0.5 degrees in radians (~55 km at mid-latitudes)
DBSCAN_EPS_RAD = 0.008727
DBSCAN_MIN_PTS = 5


# ── Data loading ──────────────────────────────────────────────────────────────

def load_firms(firms_dir: str) -> pd.DataFrame:
    """Load all FIRMS CSV files from firms_dir and return a clean DataFrame."""
    csv_files = sorted(glob.glob(os.path.join(firms_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{firms_dir}'.\n\n"
            "Download annual archive files from:\n"
            "  https://firms.modaps.eosdis.nasa.gov/data/active_fire/modis-c6.1/csv/\n\n"
            "Files to download (one per year, 2012-2018):\n"
            "  MODIS_C6_1_USA_contiguous_and_Hawaii_<YEAR>.csv\n\n"
            f"Place them in: {firms_dir}\n"
        )

    frames = []
    for path in csv_files:
        try:
            df = pd.read_csv(
                path,
                usecols=lambda c: c.lower() in {"latitude", "longitude", "acq_date", "frp"},
                low_memory=False,
            )
            df.columns = [c.lower() for c in df.columns]
            frames.append(df)
            print(f"  Loaded {len(df):>8,} rows  <-  {os.path.basename(path)}")
        except Exception as exc:
            print(f"  [skip] {os.path.basename(path)}: {exc}")

    if not frames:
        raise ValueError("All FIRMS CSV files failed to load — check file format.")

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={"acq_date": "date"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["latitude", "longitude", "date"])

    before = len(raw)
    mask = (
        (raw["longitude"] <= WESTERN_US_LON) &
        (raw["latitude"]  >= US_LAT_MIN) &
        (raw["latitude"]  <= US_LAT_MAX) &
        (raw["date"].dt.year >= DATASET_YR_MIN) &
        (raw["date"].dt.year <= DATASET_YR_MAX)
    )
    filtered = raw[mask].copy().reset_index(drop=True)
    print(f"\n[FIRMS] {before:,} total detections loaded")
    print(f"[FIRMS] {len(filtered):,} after western-US / {DATASET_YR_MIN}-{DATASET_YR_MAX} filter")
    return filtered


# ── Clustering ────────────────────────────────────────────────────────────────

def cluster_events(firms: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster FIRMS detections into fire event regions using DBSCAN
    (haversine distance, eps ~55 km).

    Returns a DataFrame with one row per cluster:
        cluster_id, lat_center, lon_center, date (earliest), n_detections
    """
    print("[FIRMS] Clustering detections into fire events ...")
    coords_rad = np.radians(firms[["latitude", "longitude"]].values)

    labels = DBSCAN(
        eps=DBSCAN_EPS_RAD,
        min_samples=DBSCAN_MIN_PTS,
        algorithm="ball_tree",
        metric="haversine",
    ).fit_predict(coords_rad)

    firms = firms.copy()
    firms["cluster_id"] = labels
    firms = firms[firms["cluster_id"] >= 0]   # drop noise (label == -1)

    events = (
        firms.groupby("cluster_id")
        .agg(
            lat_center   = ("latitude",  "mean"),
            lon_center   = ("longitude", "mean"),
            date         = ("date",      "min"),
            n_detections = ("latitude",  "count"),
        )
        .reset_index(drop=True)
    )
    print(f"[FIRMS] {len(events):,} fire event clusters identified")
    return events


# ── Sample fingerprinting ─────────────────────────────────────────────────────

def compute_fire_fractions(test_x: np.ndarray) -> np.ndarray:
    """
    Compute the fraction of active PrevFireMask pixels per sample.
    Used as a proxy for relative fire event size.
    Shape: (N,)  values in [0, 1].
    """
    prev_fire = test_x[:, PREV_FIRE_CH, :, :]    # (N, 64, 64)
    return (prev_fire > 0.5).mean(axis=(1, 2))    # (N,)


# ── Assignment ────────────────────────────────────────────────────────────────

def assign_samples(
    fire_fracs: np.ndarray,
    events: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign each test sample to a fire event cluster.

    For each sample, score every cluster by |normalised_size - fire_fraction|.
    Draw randomly from the top-10% of size-matched candidates to ensure
    geographic diversity across samples with similar fire densities.

    Returns DataFrame with columns: sample_idx, lat_center, lon_center, date.
    """
    rng      = np.random.default_rng(seed)
    n        = len(fire_fracs)
    n_events = len(events)
    det_norm = events["n_detections"].values / events["n_detections"].max()
    k        = max(1, n_events // 10)

    rows = []
    for idx in range(n):
        diff       = np.abs(det_norm - fire_fracs[idx])
        candidates = np.argpartition(diff, min(k, n_events - 1))[:k]
        chosen     = int(rng.choice(candidates))
        ev         = events.iloc[chosen]
        rows.append({
            "sample_idx": idx,
            "lat_center": round(float(ev["lat_center"]), 4),
            "lon_center": round(float(ev["lon_center"]), 4),
            "date":       ev["date"].strftime("%Y-%m-%d"),
        })

    return pd.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign approximate FIRMS coordinates to wildfire test samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--firms_dir", default="data/raw/firms",
                        help="Directory containing FIRMS MODIS C6.1 CSV files")
    parser.add_argument("--test_x",   default="data/processed/test_x.npy",
                        help="Path to test_x.npy")
    parser.add_argument("--output",   default="data/processed/sample_metadata.csv",
                        help="Output path for sample_metadata.csv")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed for reproducible assignment")
    args = parser.parse_args()

    print("=" * 60)
    print("FIRMS coordinate assignment")
    print("Coordinates are APPROXIMATE — fire-size matched, not")
    print("verified per-patch locations.")
    print("=" * 60, "\n")

    # Step 1: load and filter FIRMS data
    firms  = load_firms(args.firms_dir)

    # Step 2: cluster into fire event regions
    events = cluster_events(firms)

    # Step 3: load test samples and compute fingerprints
    print(f"\n[Test]  Loading {args.test_x} ...")
    test_x = np.load(args.test_x)
    print(f"[Test]  {test_x.shape[0]} samples, shape {test_x.shape}")
    fracs  = compute_fire_fractions(test_x)

    # Step 4: assign coordinates
    print("\n[Assign] Matching samples to fire event clusters ...")
    meta = assign_samples(fracs, events, seed=args.seed)

    # Step 5: save
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    meta.to_csv(args.output, index=False)

    unique = meta[["lat_center", "lon_center"]].drop_duplicates().shape[0]
    print(f"\n[Done]  {len(meta)} rows  ->  {args.output}")
    print(f"        {unique} unique locations across {len(meta)} samples")
    print()
    print("NOTE: Restart the Streamlit app to see updated Sample Map coordinates.")
    print("      Coordinates are approximate — see script docstring for details.")


if __name__ == "__main__":
    main()
