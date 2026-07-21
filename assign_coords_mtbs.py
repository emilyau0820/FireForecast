"""
assign_coords_mtbs.py
---------------------
Assigns APPROXIMATE geographic coordinates to each test sample by matching
against the MTBS (Monitoring Trends in Burn Severity) fire perimeter database.

IMPORTANT — coordinate accuracy
--------------------------------
The Kaggle "Next Day Wildfire Spread" TFRecords contain NO geographic
metadata.  This script distributes the 1,689 test-sample dots across real
western-US fire event centroids from the dataset's 2012-2018 period, improving
the Sample Map from "all dots at one point" to "dots spread across actual fire
zones."  Individual placements are NOT verified against the precise patch
location — treat coordinates as plausible fire-event locations, not exact
patch centroids.

Data source
-----------
MTBS (Monitoring Trends in Burn Severity) — USGS / USFS
  https://www.mtbs.gov/direct-download

The shapefile used here was downloaded from:
  https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/
      composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip

It contains ~30,000 US fire perimeters from 1984 to present.
No account or API key is required; the download is public.

Algorithm
---------
1.  Read the MTBS attribute table (DBF) from the extracted shapefile.
    No GIS library required — the DBF is parsed with the standard library.
2.  Filter to:
      • incid_type == "Wildfire"
      • western US  (lon <= -100 degrees)
      • 2012-2018   (dataset's temporal range)
3.  For each test sample, compute the fraction of active PrevFireMask pixels
    (channel 11) as a proxy for relative fire event size.
4.  Assign each sample to a randomly chosen event whose burned area (acres)
    is proportional to the sample's fire fraction (top-10% size-matched pool).
5.  Save data/processed/sample_metadata.csv with columns:
        sample_idx, lat_center, lon_center, date

Prerequisites
-------------
No third-party GIS libraries needed.  Only numpy and pandas are required.

    pip install numpy pandas         # (already needed by the rest of the project)

Usage
-----
    python assign_coords_mtbs.py
    python assign_coords_mtbs.py --mtbs_dir data/raw/mtbs --seed 42

    # Custom paths:
    python assign_coords_mtbs.py \\
        --mtbs_dir   data/raw/mtbs \\
        --test_x     data/processed/test_x.npy \\
        --output     data/processed/sample_metadata.csv \\
        --seed       42
"""

import argparse
import os
import struct

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

PREV_FIRE_CH   = 11       # PrevFireMask channel index in test_x.npy
WESTERN_US_LON = -100.0   # dataset events occur west of this longitude
US_LAT_MIN     = 24.0
US_LAT_MAX     = 50.0
DATASET_YR_MIN = 2012     # Huot et al. 2022 dataset spans 2012-2018
DATASET_YR_MAX = 2018
WILDFIRE_TYPE  = "Wildfire"


# ── DBF reader (no external GIS library required) ─────────────────────────────

def read_dbf(dbf_path: str) -> pd.DataFrame:
    """
    Read a dBASE III/IV .dbf file into a pandas DataFrame.

    Only reads non-deleted records.  Numeric fields are converted to float;
    everything else is kept as str.
    """
    with open(dbf_path, "rb") as f:
        header      = f.read(32)
        n_records   = struct.unpack("<I", header[4:8])[0]
        header_size = struct.unpack("<H", header[8:10])[0]
        record_size = struct.unpack("<H", header[10:12])[0]

        # ── Field descriptors ─────────────────────────────────────────────────
        fields = []
        f.seek(32)
        while True:
            fd = f.read(32)
            if not fd or fd[0] == 0x0D:   # header terminator
                break
            name   = fd[:11].rstrip(b"\x00").decode("ascii", errors="replace")
            ftype  = chr(fd[11])
            length = fd[16]
            fields.append((name, ftype, length))

        # ── Records ───────────────────────────────────────────────────────────
        f.seek(header_size)
        rows = []
        for _ in range(n_records):
            raw = f.read(record_size)
            if not raw:
                break
            if raw[0] == 0x2A:      # deletion flag
                continue
            rec = {}
            pos = 1
            for name, ftype, length in fields:
                raw_val = raw[pos: pos + length].decode("latin-1").strip()
                if ftype == "N" and raw_val:
                    try:
                        rec[name] = float(raw_val)
                    except ValueError:
                        rec[name] = float("nan")
                elif ftype == "N":
                    rec[name] = float("nan")
                else:
                    rec[name] = raw_val
                pos += length
            rows.append(rec)

    return pd.DataFrame(rows)


# ── MTBS loading and filtering ────────────────────────────────────────────────

def load_mtbs(mtbs_dir: str) -> pd.DataFrame:
    """
    Load the MTBS attribute table, filter to western-US wildfires, 2012-2018.

    Returns a DataFrame with columns:
        lat_center, lon_center, date, burnbndac
    """
    dbf_path = os.path.join(mtbs_dir, "mtbs_perims_DD.dbf")
    if not os.path.exists(dbf_path):
        # Try extracting from the zip if it hasn't been unpacked yet
        zip_path = os.path.join(os.path.dirname(mtbs_dir), "mtbs_perimeter_data.zip")
        if os.path.exists(zip_path):
            import zipfile
            print(f"  Extracting {zip_path} -> {mtbs_dir} ...")
            os.makedirs(mtbs_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(mtbs_dir)
        else:
            raise FileNotFoundError(
                f"MTBS DBF not found at '{dbf_path}'.\n\n"
                "Download the MTBS perimeter shapefile from:\n"
                "  https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/"
                "composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip\n\n"
                "Place mtbs_perimeter_data.zip in data/raw/, then re-run this script\n"
                "(it will extract automatically), or unzip manually to data/raw/mtbs/.\n"
            )

    print(f"[MTBS] Reading {dbf_path} ...")
    df = read_dbf(dbf_path)
    print(f"[MTBS] {len(df):,} total records loaded")

    # ── Parse coordinates and date ────────────────────────────────────────────
    df["lat_center"] = pd.to_numeric(df["burnbndlat"], errors="coerce")
    df["lon_center"] = pd.to_numeric(df["burnbndlon"], errors="coerce")
    df["date"]       = pd.to_datetime(df["ig_date"],   errors="coerce")
    df               = df.dropna(subset=["lat_center", "lon_center", "date"])

    # ── Filter ────────────────────────────────────────────────────────────────
    mask = (
        (df["incid_type"] == WILDFIRE_TYPE)             &
        (df["lon_center"] <= WESTERN_US_LON)            &
        (df["lat_center"] >= US_LAT_MIN)                &
        (df["lat_center"] <= US_LAT_MAX)                &
        (df["date"].dt.year >= DATASET_YR_MIN)          &
        (df["date"].dt.year <= DATASET_YR_MAX)
    )
    filtered = df[mask][["lat_center", "lon_center", "date", "burnbndac"]].copy()
    filtered = filtered.reset_index(drop=True)

    print(f"[MTBS] {len(filtered):,} western-US wildfires, {DATASET_YR_MIN}-{DATASET_YR_MAX}")
    return filtered


# ── Sample fingerprinting ─────────────────────────────────────────────────────

def compute_fire_fractions(test_x: np.ndarray) -> np.ndarray:
    """
    Compute the fraction of active PrevFireMask pixels per sample.
    Used as a proxy for relative fire event size.  Shape: (N,).
    """
    prev_fire = test_x[:, PREV_FIRE_CH, :, :]     # (N, 64, 64)
    return (prev_fire > 0.5).mean(axis=(1, 2))     # (N,)


# ── Assignment ────────────────────────────────────────────────────────────────

def assign_samples(
    fire_fracs: np.ndarray,
    events: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Assign each test sample to an MTBS fire event.

    Strategy: normalise burned area to [0, 1]; for each sample score every
    event by |normalised_area - fire_fraction|; draw randomly from the top-10%
    size-matched candidates to ensure geographic diversity across samples with
    similar fire densities.

    Returns DataFrame with columns: sample_idx, lat_center, lon_center, date.
    """
    rng      = np.random.default_rng(seed)
    n        = len(fire_fracs)
    n_events = len(events)

    # Normalise burned area; fill NaN with median
    areas    = events["burnbndac"].values.astype(float)
    med      = np.nanmedian(areas)
    areas    = np.where(np.isnan(areas), med, areas)
    max_area = areas.max()
    area_norm = areas / max_area if max_area > 0 else areas

    k = max(1, n_events // 10)   # top-10% pool

    rows = []
    for idx in range(n):
        diff       = np.abs(area_norm - fire_fracs[idx])
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
        description="Assign approximate MTBS coordinates to wildfire test samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mtbs_dir", default="data/raw/mtbs",
                        help="Directory containing extracted MTBS shapefile files")
    parser.add_argument("--test_x",   default="data/processed/test_x.npy",
                        help="Path to test_x.npy")
    parser.add_argument("--output",   default="data/processed/sample_metadata.csv",
                        help="Output path for sample_metadata.csv")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed for reproducible assignment")
    args = parser.parse_args()

    print("=" * 60)
    print("MTBS coordinate assignment")
    print("Coordinates are APPROXIMATE — fire-size matched, not")
    print("verified per-patch locations.")
    print("=" * 60, "\n")

    # Step 1: load MTBS events
    events = load_mtbs(args.mtbs_dir)

    # Step 2: load test samples and compute fingerprints
    print(f"\n[Test]  Loading {args.test_x} ...")
    test_x = np.load(args.test_x)
    print(f"[Test]  {test_x.shape[0]} samples, shape {test_x.shape}")
    fracs  = compute_fire_fractions(test_x)

    # Step 3: assign
    print("\n[Assign] Matching samples to MTBS fire events ...")
    meta = assign_samples(fracs, events, seed=args.seed)

    # Step 4: save
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    meta.to_csv(args.output, index=False)

    unique = meta[["lat_center", "lon_center"]].drop_duplicates().shape[0]
    print(f"\n[Done]  {len(meta)} rows  ->  {args.output}")
    print(f"        {unique} unique locations across {len(meta)} samples")
    print()
    print("NOTE: Restart the Streamlit app to see updated Sample Map coordinates.")
    print("      Coordinates are MTBS-approximate — see script docstring for details.")


if __name__ == "__main__":
    main()
