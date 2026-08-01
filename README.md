# FireForecast

**Developed by [Emily Au](https://github.com/emilyau0820)**

Next-day wildfire spread prediction using a U-Net trained on the [Next Day Wildfire Spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread) dataset (Huot et al. 2022). Given 12 environmental input channels over a 64×64 geographic patch (~24×24 km at 375 m/pixel), the model outputs a 32×32 fire probability map for the following day.

---

## Results

| Metric | Score |
|---|---|
| F1 | 0.3081 |
| AUROC | 0.8515 |
| AUPRC | 0.2822 |
| IoU | 0.1821 |

Evaluated on the 1,689-sample test set. AUPRC is the primary metric given ~1% fire pixel class imbalance (random baseline ≈ 0.012).

---

## Model

- **Architecture:** U-Net with ResNet18 encoder backbone ([segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch))
- **Input:** 12 channels × 64×64 patch (elevation, GRIDMET weather, PDSI, NDVI, ERC, population, PrevFireMask)
- **Output:** 32×32 fire spread probability map
- **Loss:** Combined Focal Loss (γ=2.0, pos\_weight=10.0) + soft Dice Loss (α=0.5 blend)
- **Training:** Adam (lr=3e-3), OneCycleLR scheduler, gradient clipping, 40 epochs

---

## Quick Start

```powershell
# 1. Clone
git clone https://github.com/emilyau0820/FireForecast.git
cd FireForecast

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. Download the dataset from Kaggle
kaggle datasets download -d fantineh/next-day-wildfire-spread
# Unzip TFRecord files into data/raw/

# 4. Preprocess raw TFRecords into NumPy arrays (one-time, ~10 min)
python preprocess.py

# 5. Train the model (~40 epochs)
python src/train.py

# 6. Evaluate and cache predictions
python src/eval.py

# 7. (Optional) Assign MTBS fire coordinates to the Sample Map
python assign_coords_mtbs.py

# 8. Launch the interactive app
streamlit run app.py
```

> For full setup details, troubleshooting, and CLI flags see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md).

---

## App Features

The Streamlit app (`app.py`) has two tabs:

**Prediction View** — interactive two-panel figure per sample:
- 3D terrain surface with fire probability painted on as colour (black → bright yellow)
- Ground-truth next-day fire mask overlaid as cyan dots
- Metrics row (max fire prob, predicted/actual burn %, mean elevation)
- 12-channel feature summary table with risk interpretation labels

**Sample Map** — overview of all 1,689 test samples:
- Dots coloured by max predicted fire probability, spread across real western-US fire locations (MTBS-approximate)
- Date range slider to filter samples by fire season
- Click any dot to select it; the Prediction View updates to show that sample
- Quick Risk Summary card (Max Fire Prob, ERC, Wind Speed, PDSI) visible without switching tabs

---

## Dataset

- **Source:** [Kaggle — Next Day Wildfire Spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread) (Huot et al. 2022)
- **Coverage:** Western United States, 2012–2018
- **Size:** 18,545 samples (14,979 train / 1,877 val / 1,689 test), ~3.8 GB raw
- **Channels:** Elevation (SRTM), wind/temp/humidity/precip/ERC/PDSI (GRIDMET), NDVI (MODIS), population (GPWv4), fire mask (VIIRS)

The dataset is static. See [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for the live-data roadmap (automated ingestion, incremental retraining, live predictions).

---

## Project Structure

```
FireForecast/
├── app.py                   # Streamlit web app
├── visualize_3d.py          # Standalone 3D Plotly CLI
├── preprocess.py            # TFRecord → NumPy pipeline
├── infer.py                 # Single-sample inference
├── assign_coords_mtbs.py    # MTBS coordinate assignment (optional)
├── assign_coords_firms.py   # FIRMS coordinate assignment (alternative)
├── src/
│   ├── train.py
│   ├── eval.py
│   ├── datasets/wildfire_dataset.py
│   ├── models/unet.py
│   └── utils/{losses,metrics}.py
├── configs/base.yaml        # All hyperparameters
├── data/                    # Populated locally — not in git (see .gitignore)
├── checkpoints/             # Model weights — distributed via GitHub Releases
└── DEVELOPER_GUIDE.md
```

---

## References

Huot, F., Hu, R. L., Goyal, N., Palard, T., Ihme, M., & Wang, Y. X. (2022). *Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading from Remote-Sensing Data*. IEEE Transactions on Geoscience and Remote Sensing.
