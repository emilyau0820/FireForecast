# Wildfire Predictor — Developer Guide

A deep learning system that predicts next-day wildfire spread as a binary
segmentation task. Given 12 environmental input channels over a 64x64 geographic
patch, the model outputs a 32x32 fire probability map showing where fire is likely
to spread the following day.

---

## Quick Start

Run these commands from the project root in order.

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Preprocess raw TFRecords into NumPy arrays (once only)
python preprocess.py

# 3. Train the model
python src/train.py

# 4. Evaluate and cache predictions
python src/eval.py

# 5a. Launch the interactive web app
streamlit run app.py

# 5b. OR generate a portable HTML visualization for a specific sample
python visualize_3d.py --idx 0

# 6. (Optional) Add approximate geographic coordinates to the Sample Map
#    MTBS data already downloaded to data/raw/mtbs/
python assign_coords_mtbs.py
```

---

## Table of Contents

1. [Tech Stack](#tech-stack)
2. [Project Structure](#project-structure)
3. [Setup and Installation](#setup-and-installation)
4. [Running the Project — Step by Step](#running-the-project--step-by-step)
5. [Feature Descriptions](#feature-descriptions)
6. [Configuration Reference](#configuration-reference)
7. [Troubleshooting](#troubleshooting)

---

## Tech Stack

| Layer | Library | Purpose |
|---|---|---|
| Deep learning | PyTorch 2.1+, segmentation-models-pytorch | U-Net model training and inference |
| Data pipeline | TensorFlow 2.12+ | Reading raw TFRecord files from Kaggle |
| Numerics | NumPy, pandas | Array operations, data handling |
| Geospatial | rasterio, xarray | Raster I/O (for future GeoTIFF export) |
| Visualization | Plotly 5.18+, matplotlib, seaborn | 3D figures, training plots |
| App | Streamlit | Interactive web UI |
| Experiment tracking | Weights & Biases (W&B) | Training metrics and run logging |
| ML utilities | scikit-learn | Metrics (F1, AUROC, AUPRC, IoU) |
| Config | PyYAML | YAML config parsing |

---

## Project Structure

```
Wildfire Predictor/
|
|-- app.py                      <- Streamlit web app (main entry point for UI)
|-- visualize_3d.py             <- 3D Plotly visualization; standalone CLI tool
|-- preprocess.py               <- TFRecord -> NumPy pipeline
|-- infer.py                    <- Single-sample inference with 12-channel plot
|-- assign_coords_mtbs.py       <- Assigns approximate coordinates from MTBS fire perimeters (optional)
|-- assign_coords_firms.py      <- Alternative: assigns coordinates from NASA FIRMS CSVs (optional)
|
|-- src/
|   |-- train.py                <- Training loop (run from project root)
|   |-- eval.py                 <- Test set evaluation + saves predictions.npy
|   |-- datasets/
|   |   `-- wildfire_dataset.py <- PyTorch Dataset; per-channel normalization
|   |-- models/
|   |   `-- unet.py             <- U-Net builder (segmentation-models-pytorch)
|   `-- utils/
|       |-- losses.py           <- BCE, Focal, Combined+Dice loss functions
|       `-- metrics.py          <- F1, AUROC, AUPRC, IoU
|
|-- configs/
|   `-- base.yaml               <- All hyperparameters, paths, and geo settings
|
|-- data/
|   |-- raw/                    <- TFRecord files downloaded from Kaggle (19 files)
|   |-- raw/
|   |   |-- mtbs/               <- MTBS perimeter shapefile (extracted; used by assign_coords_mtbs.py)
|   |   |-- mtbs_perimeter_data.zip <- Source zip (downloaded automatically on first run)
|   |   `-- firms/              <- FIRMS MODIS C6.1 CSVs (optional; used by assign_coords_firms.py)
|   `-- processed/              <- NumPy arrays produced by preprocess.py
|       |-- train_x.npy / train_y.npy
|       |-- val_x.npy   / val_y.npy
|       |-- test_x.npy  / test_y.npy
|       |-- predictions.npy     <- Saved by eval.py; consumed by app.py and visualize_3d.py
|       `-- sample_metadata.csv <- Per-sample lat/lon/date; default coords until MTBS step is run
|
|-- checkpoints/
|   `-- best_model.pth          <- Best checkpoint saved during training
|
|-- outputs/                    <- PNG plots from eval.py; HTML files from visualize_3d.py
|-- wandb/                      <- W&B run logs (auto-generated during training)
|-- .streamlit/
|   `-- config.toml             <- Dark theme and font settings for the Streamlit app
|-- requirements.txt
`-- DEVELOPER_GUIDE.md          <- This file
```

---

## Setup and Installation

### Prerequisites

- Python 3.10 or later
- Anaconda or a virtual environment (recommended)
- A GPU is optional but significantly speeds up training and inference

### 1. Clone and enter the project

```powershell
cd "C:\Users\emily\Personal Projects\Wildfire Predictor"
```

### 2. Install dependencies

Use `python -m pip` rather than calling `pip` directly to avoid the
"Fatal error in launcher" issue on Windows when the venv launcher path
is stale.

```powershell
python -m pip install -r requirements.txt
```

> **Note:** `tensorflow` is listed in `requirements.txt` but is only used by
> `preprocess.py` to read the raw TFRecord files. If you already have the
> processed `.npy` files in `data/processed/`, TensorFlow is not needed
> for any other step.

### 3. Download the dataset (first time only)

The raw data comes from the Kaggle *Next Day Wildfire Spread* dataset.

```powershell
kaggle datasets download -d fantineh/next-day-wildfire-spread
```

Unzip it into `data/raw/`. You should see files like:
`next_day_wildfire_spread_train_00.tfrecord`, etc.

---

## Running the Project — Step by Step

Run every command from the **project root** directory
(`C:\Users\emily\Personal Projects\Wildfire Predictor`).

### Step 1 — Preprocess the raw data

Converts TFRecord files into NumPy arrays. Only needs to be run once.

```powershell
python preprocess.py
```

**Output:** `data/processed/train_x.npy`, `val_x.npy`, `test_x.npy`, and
corresponding `_y.npy` label files.

---

### Step 2 — Train the model

```powershell
python src/train.py
```

Trains for 40 epochs (configurable in `configs/base.yaml`). Saves the best
checkpoint to `checkpoints/best_model.pth` based on validation AUPRC.
Logs metrics to Weights & Biases — set your W&B username in `configs/base.yaml`
under `logging.entity` before running.

**Output:** `checkpoints/best_model.pth`

---

### Step 3 — Evaluate the model

```powershell
python src/eval.py
```

Runs the checkpoint on the test set. Prints F1, AUROC, AUPRC, and IoU.
Saves 10 PNG visualizations to `outputs/` and caches all test predictions
to `data/processed/predictions.npy` for use by the app and `visualize_3d.py`.

**Output:** `data/processed/predictions.npy`, `outputs/sample_0000.png` ... `sample_0009.png`

---

### Step 4a — Launch the Streamlit app

The main interactive UI. Requires `data/processed/predictions.npy` (from Step 3).

```powershell
streamlit run app.py
```

Opens automatically at `http://localhost:8501`. See the
[Feature Descriptions](#feature-descriptions) section below for a full walkthrough.

---

### Step 4b — Generate a standalone 3D visualization

Produces a self-contained HTML file you can open offline or share by email.
Generates `predictions.npy` automatically if it does not exist.

```powershell
# Basic usage — sample 0, default California coordinates
python visualize_3d.py

# Choose a different sample
python visualize_3d.py --idx 5

# Place the map overlay on a specific location
python visualize_3d.py --idx 3 --lat 34.1 --lon -118.2

# Save without opening a browser tab
python visualize_3d.py --idx 0 --no-browser

# Specify a custom output path
python visualize_3d.py --idx 0 --output results/my_map.html
```

**Output:** `outputs/viz_3d_sample_<idx>.html`

---

### Step 4c — Single-sample inference with all 12 channels

Visualizes all 12 input channels alongside the model prediction.

```powershell
python infer.py --idx 42
```

**Output:** `outputs/infer_sample_42.png`

---

### Step 5 — Add approximate geographic coordinates (optional)

The Kaggle TFRecords contain no per-patch coordinates. This step assigns
approximate locations from the **MTBS (Monitoring Trends in Burn Severity)**
fire perimeter database, distributing the 1,689 test-sample dots across real
western-US wildfire events from the dataset's 2012–2018 period.

> **Accuracy note:** Coordinates are matched by relative fire-event size
> (PrevFireMask pixel fraction), not by verified patch location. They represent
> plausible wildfire centroids in the western US, not exact patch origins.

#### 5a — The MTBS data

The MTBS perimeter shapefile has already been downloaded to `data/raw/mtbs/`
and extracted during the initial project setup. It is sourced from USGS/USFS
at no cost and requires no account:

```
https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/MTBS_Fire/data/
    composite_data/burned_area_extent_shapefile/mtbs_perimeter_data.zip
```

If the `data/raw/mtbs/` directory is missing, the script will automatically
re-extract it from `data/raw/mtbs_perimeter_data.zip`. If that zip is also
missing, re-download it from the URL above and place it in `data/raw/`.

#### 5b — Run the assignment script

```powershell
python assign_coords_mtbs.py
```

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--mtbs_dir` | `data/raw/mtbs` | Directory containing extracted MTBS `.dbf` |
| `--test_x` | `data/processed/test_x.npy` | Test set input array |
| `--output` | `data/processed/sample_metadata.csv` | Output metadata path |
| `--seed` | `42` | Random seed for reproducible assignment |

**Output:** `data/processed/sample_metadata.csv`
(878 unique fire locations across 1,689 samples)

Once this file has varied coordinates (more than one unique lat/lon), the
Streamlit app automatically:
- Spreads the Sample Map dots across real fire locations
- Centres the map on each selected sample (zoom 6)
- Auto-fills the sidebar lat/lon inputs with the sample's coordinates
- Shows the selected sample as a yellow-highlighted dot

#### Alternative: FIRMS-based assignment

`assign_coords_firms.py` is a second assignment script that uses NASA FIRMS
MODIS C6.1 point detections instead of MTBS perimeters, and additionally runs
DBSCAN spatial clustering. To use it, you must supply your own FIRMS CSV files
(annual archive files are not available for direct download; a MAP_KEY from
`https://firms.modaps.eosdis.nasa.gov/usfs/api/map_key/` is required for the
FIRMS API). MTBS is recommended for most uses.

---

---

## Feature Descriptions

### Streamlit App (`app.py`)

The app has two tabs — **Prediction View** (the 3D visualisation and detail
panel) and **Sample Map** (an overview map of all test samples). All sidebar
controls apply globally and update both tabs.

#### Sidebar controls

| Control | What it does |
|---|---|
| **Sample Index slider** | Selects which of the 1,689 test samples to display. Bound to `st.session_state.selected_idx` so that clicking a point on the Sample Map automatically updates it. |
| **Jump to index (text box)** | Type a sample number and press Enter to jump directly. Shows a red error for negative values, values above the maximum, or non-numeric text. |
| **Latitude / Longitude inputs** | Sets the geographic centre for the right-panel map in the Prediction View. Auto-fills with the selected sample's MTBS coordinates when available. |
| **Fire Probability Threshold** | Pixels above this value count as "on fire" in the metrics row. The 3D colour gradient always shows the continuous probability regardless of threshold. |
| **Show ground truth checkbox** | When checked, renders the actual next-day fire mask as cyan dots 60 m above the terrain surface in the Prediction View. |
| **Date Range Filter (Fire season slider)** | Appears when `sample_metadata.csv` has a populated date column (requires `assign_coords_mtbs.py`). Drag either handle to narrow the Sample Map to a specific fire-season window. The caption below the map shows how many samples are currently visible. The Prediction View and Quick Risk Summary always reflect the selected sample regardless of this filter. |

#### Prediction View tab

**Chart:** Two panels in a single interactive Plotly figure:
- **Left — 3D Terrain:** Elevation extruded as a 3D surface, fire probability
  painted on as colour (black at 0 %, bright yellow at 100 %). Orange markers
  float above currently-burning pixels. Rotate with left-click-drag, zoom with
  scroll wheel.
- **Right — Geographic Map:** OpenStreetMap tiles with a fire probability heat
  overlay. Pan and zoom independently.

**Metrics row:** Four live numbers for the selected sample:

| Metric | Definition |
|---|---|
| **Max Predicted Probability** | Highest fire probability across the 32x32 patch |
| **Patch Predicted on Fire** | % of pixels >= threshold |
| **Patch Actually on Fire** | % of pixels in the ground-truth next-day mask |
| **Mean Elevation** | Average terrain height in metres |

**Sample Feature Summary:** Collapsible table (expanded by default) showing all
12 input channels computed over the full 64x64 patch with Mean, Min, Max, and a
risk interpretation label:

| Channel | Interpretation thresholds |
|---|---|
| Wind Speed | < 2 m/s = Low, 2-5 = Moderate, 5-10 = High, > 10 = Extreme |
| Max Temp (tmmx) | < 290 K = Cool, 290-305 = Warm, > 305 K = Hot |
| PDSI | > 0 = Moist, -2 to 0 = Near-Normal, -4 to -2 = Drought, < -4 = Severe Drought |
| NDVI | > 0.5 = Dense vegetation, 0.2-0.5 = Sparse, < 0.2 = Dry/Bare |
| ERC | < 30 = Low, 30-60 = Moderate, 60-90 = High, > 90 = Extreme |
| All others | N/A |

#### Sample Map tab

An overview map showing all 1,689 test samples as dots coloured by each
sample's maximum predicted fire probability (same fire colorscale as the 3D
view — black to bright yellow). The currently selected sample is highlighted
with a larger marker.

**Clicking a dot** triggers a rerun, updates `st.session_state.selected_idx`
to that sample's index, and updates the slider. A caption below the map shows
how many samples are currently visible and the selected sample's coordinates.

**Date filter:** When `sample_metadata.csv` contains a populated `date` column,
the sidebar shows a "Fire season" range slider. Narrowing the window hides
out-of-range dots from the map and updates the caption to show
`N of 1689 samples shown`. The Prediction View and Quick Risk Summary are
unaffected by the date filter — they always show the selected sample.

**Coordinate-aware behaviour:** The app detects whether `sample_metadata.csv`
contains varied coordinates (more than one unique lat/lon). If so:
- The Sample Map spreads dots across real fire locations and centres on the
  selected sample (zoom 6)
- The selected sample is highlighted with a yellow dot inside a white ring
- The sidebar lat/lon inputs auto-fill with the sample's MTBS coordinates,
  keeping the 3D terrain surface and the 2D map overlay correctly positioned
- A caption below the map notes that coordinates are MTBS-approximate

If coordinates are all-default (or the file is absent), an info banner
explains how to run `assign_coords_mtbs.py` to populate them.

**Quick Risk Summary:** An expandable card immediately below the map (expanded
by default) that shows four key risk indicators for the currently selected
sample without requiring a tab switch:

| Metric | Source | Thresholds |
|---|---|---|
| **Max Fire Prob** | `predictions[idx].max()` | — |
| **ERC** | Channel 9 mean | < 30 Low, 30-60 Moderate, 60-90 High, > 90 Extreme |
| **Wind Speed** | Channel 2 mean | < 2 m/s Low, 2-5 Moderate, 5-10 High, > 10 Extreme |
| **PDSI** | Channel 7 mean | > 0 Moist, -2 to 0 Near-Normal, -4 to -2 Drought, < -4 Severe |

The risk category (e.g. "High", "Drought") appears as a neutral sub-label
below each value using Streamlit's `delta_color="off"` parameter.

**sample_metadata.csv format:**

| Column | Type | Description |
|---|---|---|
| `sample_idx` | int | Row index matching `test_x.npy` |
| `lat_center` | float | Patch centre latitude |
| `lon_center` | float | Patch centre longitude |
| `date` | str or None | Sample date (YYYY-MM-DD) if available |

---

### Metadata Discovery (`preprocess.py` — `extract_metadata`)

`extract_metadata(tfrecord_path)` inspects the first example in a TFRecord file
and prints every feature key it finds, then highlights any whose names contain
geographic or temporal keywords (`date`, `time`, `year`, `month`, `day`, `lat`,
`lon`, `location`). It does not modify any data.

Run it once after downloading the raw dataset to see what metadata is available:

```powershell
python -c "from preprocess import extract_metadata; extract_metadata('data/raw/next_day_wildfire_spread_train_00.tfrecord')"
```

Actual output from `next_day_wildfire_spread_train_00.tfrecord`:
```
All feature keys in first record of '...train_00.tfrecord'
  Total: 13
  FireMask
  NDVI
  PrevFireMask
  elevation
  erc
  pdsi
  population
  pr
  sph
  th
  tmmn
  tmmx
  vs

Keys matching metadata keywords {'lat', 'date', 'year', 'day', 'time', 'month', 'location', 'lon'}:
  [float_list] population: [19.67, 18.79, ...]   <- false positive (substrings "lat"/"lon")
```

**The Kaggle TFRecords do not include per-sample geographic coordinates or dates.**
The 13 keys are exactly the 12 input features and the `FireMask` label. `population`
matched only because it contains the letters "lat" as a substring.

Until `assign_coords_mtbs.py` is run (see Step 5), `sample_metadata.csv`
holds the default config centre (37.5 N, 119.5 W) for all samples, so all
1,689 dots overlap at one point on the Sample Map.

---

### 3D Visualization CLI (`visualize_3d.py`)

Standalone script that produces the same two-panel Plotly figure as the app
but saves it as a portable HTML file. Useful for sharing results or for
batch-generating visualizations without running the full Streamlit server.

Automatically generates and caches `data/processed/predictions.npy` on first
run if it does not already exist, by loading the checkpoint and running
inference over the full test set.

**CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--idx` | `0` | Test sample index to visualize |
| `--config` | `configs/base.yaml` | Path to YAML configuration file |
| `--predictions` | auto-detected | Path to a non-default `predictions.npy` file |
| `--lat` | from config | Patch centre latitude (overrides `geo.lat_center`) |
| `--lon` | from config | Patch centre longitude (overrides `geo.lon_center`) |
| `--output` | `outputs/viz_3d_sample_<idx>.html` | Where to save the HTML file |
| `--no-browser` | off | Save without opening a browser tab |

---

## Configuration Reference (`configs/base.yaml`)

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"

model:
  encoder: "resnet18"        # Backbone — change to "resnet34" or "efficientnet-b2" to experiment
  in_channels: 12
  out_channels: 1

training:
  epochs: 40
  batch_size: 32
  learning_rate: 3.0e-3
  pos_weight: 10.0           # Upweights fire pixels to handle class imbalance
  focal_gamma: 2.0
  random_crop_size: 32

eval:
  checkpoint: "checkpoints/best_model.pth"
  output_dir: "outputs/"
  threshold: 0.5

geo:
  lat_center: 37.5           # Default map placement latitude
  lon_center: -119.5         # Default map placement longitude
  meters_per_pixel: 375.0    # VIIRS nominal resolution
```

---

## Troubleshooting

### "Fatal error in launcher" when running pip

Do not call `pip` directly on Windows if the venv launcher is stale.
Use `python -m pip` instead:

```powershell
python -m pip install -r requirements.txt
python -m pip install plotly
```

### App shows a blank chart

`data/processed/predictions.npy` is missing. Run `python src/eval.py` first,
or run `python visualize_3d.py` once (it generates and caches predictions
automatically).

### Unicode errors on Windows terminal

Some terminals use the cp1252 encoding by default. Set UTF-8 for the session:

```powershell
$env:PYTHONUTF8 = "1"
streamlit run app.py
```

Or set it permanently in your Anaconda environment.

### Streamlit app is light-themed

The dark theme is set in `.streamlit/config.toml`. If it is not applying,
confirm the file exists in the project root and restart the Streamlit server.

### W&B login prompt during training

Run `wandb login` once in your terminal before training, or disable W&B
entirely by setting the environment variable:

```powershell
$env:WANDB_MODE = "disabled"
python src/train.py
```

### "Widget created with default value but also set via Session State API"

This warning appears if a slider or other widget is given both `key=` and
`value=` while the matching session state key already exists. The fix (already
applied in `app.py`) is to pass only `key=` and let Streamlit read the initial
value from `st.session_state` automatically:

```python
# Wrong — causes the warning
st.slider("Sample Index", 0, n-1, value=st.session_state.selected_idx, key="selected_idx")

# Correct — key= alone is sufficient
st.slider("Sample Index", 0, n-1, key="selected_idx")
```

### "The value 'i' is not a valid emoji" in st.info / st.warning / st.error

Streamlit's `icon=` parameter requires a single Unicode emoji character such as
`"!"` or an actual emoji like `"[fire]"`. Shortcode strings like `"i"`, `"info"`,
or `":information_source:"` are not accepted. Either pass a single character or
omit `icon=` entirely to use the default icon for that alert type.

