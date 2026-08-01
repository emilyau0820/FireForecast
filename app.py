"""
app.py
------
Streamlit app for interactive wildfire spread prediction visualization.

Run from the project root:
    streamlit run app.py
"""

import datetime  # noqa: F401 — used implicitly by st.slider date type
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

# Allow imports from src/ (models, datasets, utils)
sys.path.insert(0, "src")

# visualize_3d.py lives in the project root alongside this file
from visualize_3d import FIRE_COLORSCALE, build_figure, make_latlon_grid  # noqa: E402

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    layout="wide",
    page_title="FireForecast",
    page_icon="fire",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0

# Streamlit forbids writing to a widget-owned session state key after that
# widget has been instantiated.  Map clicks therefore stage their target index
# in a neutral key (_pending_map_click) and we consume it here — before any
# widget is created — so the slider picks up the new value cleanly.
if "_pending_map_click" in st.session_state:
    st.session_state["selected_idx"] = st.session_state.pop("_pending_map_click")

# ── Data loading (cached — runs once per server lifetime) ─────────────────────

@st.cache_resource
def load_data():
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)
    test_x      = np.load("data/processed/test_x.npy")       # (N, 12, 64, 64)
    test_y      = np.load("data/processed/test_y.npy")       # (N,  1, 64, 64)
    predictions = np.load("data/processed/predictions.npy")  # (N, 32, 32)
    return config, test_x, test_y, predictions


@st.cache_resource
def load_metadata(n_samples: int, default_lat: float, default_lon: float) -> pd.DataFrame:
    """Load sample_metadata.csv if it exists; otherwise return a stub with default coords."""
    meta_path = "data/processed/sample_metadata.csv"
    if os.path.exists(meta_path):
        df = pd.read_csv(meta_path)
        # Ensure sample_idx is contiguous 0..N-1; fill gaps with defaults
        full = pd.DataFrame({"sample_idx": range(n_samples)})
        df   = full.merge(df, on="sample_idx", how="left")
        df["lat_center"] = df["lat_center"].fillna(default_lat)
        df["lon_center"] = df["lon_center"].fillna(default_lon)
        return df
    return pd.DataFrame({
        "sample_idx": range(n_samples),
        "lat_center": default_lat,
        "lon_center": default_lon,
        "date":       None,
    })


config, test_x, test_y, predictions = load_data()

geo       = config.get("geo", {})
n_samples = len(test_x)

_default_cfg_lat = float(geo.get("lat_center", 37.5))
_default_cfg_lon = float(geo.get("lon_center", -119.5))

metadata = load_metadata(n_samples, _default_cfg_lat, _default_cfg_lon)

# True when sample_metadata.csv has varied coordinates (i.e. real or MTBS-approximate,
# not all stacked at the config default centre).
has_real_coords = metadata["lat_center"].nunique() > 1

# True when the date column is meaningfully populated (not all None/NaT).
# If so, parse it once here and cache the min/max for the sidebar slider.
has_dates  = False
_min_date  = None
_max_date  = None
if "date" in metadata.columns:
    _dates_parsed = pd.to_datetime(metadata["date"], errors="coerce")
    if _dates_parsed.notna().any():
        metadata = metadata.copy()          # avoid mutating the cached DataFrame
        metadata["date_parsed"] = _dates_parsed
        has_dates  = True
        _min_date  = _dates_parsed.dropna().dt.date.min()
        _max_date  = _dates_parsed.dropna().dt.date.max()

# Per-sample max predicted probability — used to colour the Sample Map
max_probs = predictions.max(axis=(1, 2))  # (N,)

# ── Channel metadata and risk interpretation ──────────────────────────────────

# (display name, unit, internal key used by _risk_label)
CHANNEL_INFO = [
    ("Elevation",      "m",           "elevation"),
    ("Wind Direction", "degrees",     "wind_dir"),
    ("Wind Speed",     "m/s",         "wind_speed"),
    ("Min Temp",       "K",           "tmmn"),
    ("Max Temp",       "K",           "tmmx"),
    ("Humidity",       "kg/kg",       "humidity"),
    ("Precipitation",  "mm",          "precipitation"),
    ("PDSI",           "unitless",    "PDSI"),
    ("NDVI",           "0-1",         "NDVI"),
    ("ERC",            "unitless",    "ERC"),
    ("Population",     "persons/km2", "population"),
    ("Prev Fire Mask", "0/1",         "prev_fire_mask"),
]


def _risk_label(key: str, mean_val: float) -> str:
    if key == "wind_speed":
        if mean_val < 2:   return "Low"
        if mean_val < 5:   return "Moderate"
        if mean_val < 10:  return "High"
        return "Extreme"
    if key == "tmmx":
        if mean_val < 290: return "Cool"
        if mean_val < 305: return "Warm"
        return "Hot"
    if key == "PDSI":
        if mean_val > 0:   return "Moist"
        if mean_val > -2:  return "Near-Normal"
        if mean_val > -4:  return "Drought"
        return "Severe Drought"
    if key == "NDVI":
        if mean_val > 0.5: return "Dense vegetation"
        if mean_val > 0.2: return "Sparse"
        return "Dry/Bare"
    if key == "ERC":
        if mean_val < 30:  return "Low"
        if mean_val < 60:  return "Moderate"
        if mean_val < 90:  return "High"
        return "Extreme"
    return "N/A"


def _sample_coords(idx: int) -> tuple[float, float]:
    """Return (lat, lon) for a sample index, from metadata if real, else config default."""
    if has_real_coords:
        row = metadata[metadata["sample_idx"] == idx]
        if len(row) > 0:
            return float(row["lat_center"].iloc[0]), float(row["lon_center"].iloc[0])
    return _default_cfg_lat, _default_cfg_lon


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("FireForecast")
    st.caption("Predicting next-day wildfire spread using a U-Net — developed by [@emilyau0820](https://github.com/emilyau0820)")
    st.divider()

    # key="selected_idx" binds the slider to st.session_state.selected_idx.
    # Do NOT also pass value= — that causes a Streamlit conflict warning.
    sample_idx = st.slider(
        "Sample Index",
        min_value=0,
        max_value=n_samples - 1,
        key="selected_idx",
        help=f"Select one of the {n_samples} test samples",
    )

    manual_input = st.text_input(
        "Jump to index",
        value="",
        placeholder=f"Type a number (0 - {n_samples - 1})",
    )
    if manual_input.strip():
        try:
            manual_idx = int(manual_input.strip())
            if manual_idx < 0:
                st.error("Index cannot be negative.")
            elif manual_idx >= n_samples:
                st.error(f"Index too high — maximum is {n_samples - 1}.")
            else:
                sample_idx = manual_idx
                st.session_state.selected_idx = manual_idx
        except ValueError:
            st.error("Please enter a whole number.")

    st.divider()

    # Resolve default lat/lon: use per-sample metadata when real coords exist,
    # otherwise fall back to the config centre.  Passing value= here means the
    # inputs reset to the sample's coordinates whenever the sample changes,
    # while still allowing temporary manual overrides within a single session.
    _meta_lat, _meta_lon = _sample_coords(sample_idx)

    st.markdown("**Geographic Placement**")
    if has_real_coords:
        st.caption(
            "Showing MTBS-approximate coordinates for this sample. "
            "Override below if needed."
        )
    else:
        st.caption(
            "No real coordinates available. Run assign_coords_mtbs.py "
            "to populate approximate locations."
        )

    lat_center = st.number_input(
        "Latitude (centre)",
        value=_meta_lat,
        format="%.4f",
        step=0.1,
    )
    lon_center = st.number_input(
        "Longitude (centre)",
        value=_meta_lon,
        format="%.4f",
        step=0.1,
    )

    st.divider()

    threshold = st.slider(
        "Fire Probability Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Pixels above this value are counted as 'on fire' for the metrics row",
    )

    show_gt = st.checkbox(
        "Show ground truth",
        value=True,
        help="Overlay the actual next-day fire mask (cyan dots) on the 3D surface",
    )

    st.divider()

    if has_dates:
        st.markdown("**Date Range Filter**")
        st.caption("Filters the Sample Map to a fire-season window.")
        date_range = st.slider(
            "Fire season",
            min_value=_min_date,
            max_value=_max_date,
            value=(_min_date, _max_date),
            format="YYYY-MM-DD",
            help=(
                "Only samples whose MTBS ignition date falls within this window "
                "are shown on the Sample Map. The Prediction View and risk card "
                "always reflect the currently selected sample regardless of this filter."
            ),
        )
    else:
        date_range = None

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_pred, tab_map = st.tabs(["Prediction View", "Sample Map"])

# ── Prediction View ───────────────────────────────────────────────────────────

with tab_pred:

    # Center-crop 64x64 -> 32x32 to align with the 32x32 model predictions
    CROP = slice(16, 48)

    elevation  = test_x[sample_idx, 0,  CROP, CROP].astype(np.float32)  # metres
    prev_fire  = test_x[sample_idx, 11, CROP, CROP].astype(np.float32)  # 0/1
    fire_prob  = predictions[sample_idx].astype(np.float32)              # 0-1

    gt_raw       = test_y[sample_idx, 0, CROP, CROP].astype(np.float32)
    ground_truth = np.where(gt_raw > 0.5, 1.0, 0.0).astype(np.float32) if show_gt else None

    meters_per_pixel = float(geo.get("meters_per_pixel", 375.0))
    lat_grid, lon_grid = make_latlon_grid(
        lat_center, lon_center,
        grid_size=32,
        meters_per_pixel=meters_per_pixel,
    )

    st.markdown(
        f"<h3 style='text-align:center'>Wildfire Spread Prediction - Sample {sample_idx}</h3>",
        unsafe_allow_html=True,
    )

    fig = build_figure(
        elevation, fire_prob, prev_fire, ground_truth,
        lat_grid, lon_grid,
        sample_idx,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Metrics row ───────────────────────────────────────────────────────────

    max_prob       = float(fire_prob.max())
    pct_predicted  = float((fire_prob >= threshold).mean() * 100)
    pct_actual     = float((gt_raw > 0.5).mean() * 100)
    mean_elev      = float(elevation.mean())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Predicted Probability", f"{max_prob:.1%}")
    col2.metric(
        "Patch Predicted on Fire",
        f"{pct_predicted:.1f}%",
        help=f"Fraction of 32x32 pixels >= threshold {threshold:.2f}",
    )
    col3.metric(
        "Patch Actually on Fire",
        f"{pct_actual:.1f}%",
        help="Fraction of 32x32 pixels in the ground-truth next-day mask",
    )
    col4.metric("Mean Elevation", f"{mean_elev:.0f} m")

    # ── Sample Feature Summary ────────────────────────────────────────────────

    with st.expander("Sample Feature Summary", expanded=True):
        rows = []
        for ch_idx, (display_name, unit, key) in enumerate(CHANNEL_INFO):
            channel  = test_x[sample_idx, ch_idx, :, :]  # full 64x64, natural units
            mean_val = float(channel.mean())
            rows.append({
                "Feature":             f"{display_name} ({unit})",
                "Mean":                mean_val,
                "Min":                 float(channel.min()),
                "Max":                 float(channel.max()),
                "Risk Interpretation": _risk_label(key, mean_val),
            })

        df = pd.DataFrame(rows)
        st.dataframe(
            df.style.format({"Mean": "{:.2f}", "Min": "{:.2f}", "Max": "{:.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

# ── Sample Map ────────────────────────────────────────────────────────────────

with tab_map:

    # ── Coordinate quality banner ─────────────────────────────────────────────
    if not has_real_coords:
        st.info(
            "All samples are plotted at the default centre — no per-sample "
            "coordinates are available yet. "
            "Run assign_coords_mtbs.py to assign approximate locations from "
            "the MTBS fire perimeter database (see Developer Guide for instructions)."
        )
    else:
        st.caption(
            "Coordinates are MTBS-approximate (fire-size matched to real "
            "western-US fire events, 2012-2018). Not verified per-patch locations."
        )

    # ── Date filter ───────────────────────────────────────────────────────────
    # date_range is None when has_dates is False (defined in sidebar block above).
    if has_dates and date_range is not None:
        _d0, _d1 = date_range
        _vis_mask = (
            (metadata["date_parsed"].dt.date >= _d0) &
            (metadata["date_parsed"].dt.date <= _d1)
        )
        _vis = metadata[_vis_mask]
    else:
        _vis = metadata

    map_lats      = _vis["lat_center"].tolist()
    map_lons      = _vis["lon_center"].tolist()
    sample_ids    = _vis["sample_idx"].astype(int).tolist()
    vis_max_probs = max_probs[_vis["sample_idx"].values]
    n_visible     = len(_vis)

    # Selected sample coordinates for highlight and map centering
    sel_lat, sel_lon = _sample_coords(sample_idx)

    # ── Base scatter (all samples, coloured by max fire probability) ──────────
    map_fig = go.Figure(go.Scattermapbox(
        lat=map_lats,
        lon=map_lons,
        mode="markers",
        marker=dict(
            size=10,
            opacity=0.8,
            color=vis_max_probs.tolist(),
            colorscale=FIRE_COLORSCALE,
            cmin=0.0,
            cmax=1.0,
            colorbar=dict(
                title="Max Fire Prob",
                thickness=15,
                tickformat=".0%",
                x=1.0,
            ),
        ),
        customdata=sample_ids,
        hovertemplate=(
            "Sample %{customdata}<br>"
            "Lat: %{lat:.4f}  Lon: %{lon:.4f}<br>"
            "Max prob: %{marker.color:.1%}"
            "<extra></extra>"
        ),
        # selectedpoints intentionally omitted: it dims every unselected point
        # to ~20% opacity, which hides stacked dots at the same coordinates.
    ))

    # ── Selected sample highlight (white ring + yellow fill) ─────────────────
    # Two overlapping traces create a visible ring even against the fire colorscale.
    map_fig.add_trace(go.Scattermapbox(
        lat=[sel_lat],
        lon=[sel_lon],
        mode="markers",
        marker=dict(size=22, color="white", opacity=1.0),
        hoverinfo="skip",
        showlegend=False,
    ))
    map_fig.add_trace(go.Scattermapbox(
        lat=[sel_lat],
        lon=[sel_lon],
        mode="markers",
        marker=dict(size=14, color="#ffdd00", opacity=1.0),
        customdata=[sample_idx],
        hovertemplate="Selected: Sample %{customdata}<extra></extra>",
        showlegend=False,
    ))

    # ── Layout ────────────────────────────────────────────────────────────────
    # When real coords exist, centre the map on the selected sample at zoom 6;
    # otherwise centre on the dataset mean and zoom out.
    if has_real_coords:
        map_center = dict(lat=sel_lat, lon=sel_lon)
        map_zoom   = 6
    else:
        map_center = dict(
            lat=float(metadata["lat_center"].mean()),
            lon=float(metadata["lon_center"].mean()),
        )
        map_zoom = 5

    map_fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=map_center,
            zoom=map_zoom,
        ),
        height=600,
        autosize=False,          # prevents unrelated reruns from resizing the map
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="#0d0d0d",
        font=dict(color="#dddddd"),
    )

    selection = st.plotly_chart(
        map_fig,
        use_container_width=True,
        on_select="rerun",
        key="sample_map_chart",
    )

    # Handle point click.  We cannot write directly to st.session_state.selected_idx
    # here because the slider with key="selected_idx" has already been instantiated
    # this run.  Instead we stage the value in a neutral key; the resolver block at
    # the very top of the script will apply it on the next rerun before any widget
    # is created, so the slider updates cleanly.
    if selection and selection.selection and selection.selection.points:
        pt = selection.selection.points[0]
        clicked_sample = int(pt.get("customdata", pt.get("point_index", 0)))
        if clicked_sample != st.session_state.selected_idx:
            st.session_state["_pending_map_click"] = clicked_sample
            st.rerun()

    _vis_label = (
        f"{n_visible} of {n_samples} samples shown"
        if (has_dates and date_range is not None and n_visible < n_samples)
        else f"{n_samples} samples"
    )
    st.caption(
        f"Currently selected: Sample {st.session_state.selected_idx}  "
        f"({sel_lat:.4f} N, {sel_lon:.4f} E)  |  {_vis_label}  |  "
        "Click a point to update the selection."
    )

    # ── Quick Risk Summary ─────────────────────────────────────────────────────
    with st.expander(f"Quick Risk Summary — Sample {sample_idx}", expanded=True):
        _prob_val = float(predictions[sample_idx].max())
        _erc_val  = float(test_x[sample_idx, 9, :, :].mean())   # ERC        ch 9
        _wind_val = float(test_x[sample_idx, 2, :, :].mean())   # Wind Speed ch 2
        _pdsi_val = float(test_x[sample_idx, 7, :, :].mean())   # PDSI       ch 7

        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric(
            "Max Fire Prob",
            f"{_prob_val:.1%}",
            help="Highest predicted fire-spread probability across the 32×32 patch",
        )
        rc2.metric(
            "ERC (Energy Release Component)",
            f"{_erc_val:.1f}",
            delta=_risk_label("ERC", _erc_val),
            delta_color="off",
            help="Mean ERC over the 64×64 input patch. Higher → more combustible fuels.",
        )
        rc3.metric(
            "Wind Speed",
            f"{_wind_val:.1f} m/s",
            delta=_risk_label("wind_speed", _wind_val),
            delta_color="off",
            help="Mean wind speed over the 64×64 input patch.",
        )
        rc4.metric(
            "PDSI (Drought Index)",
            f"{_pdsi_val:.2f}",
            delta=_risk_label("PDSI", _pdsi_val),
            delta_color="off",
            help="Palmer Drought Severity Index. Negative → dry conditions.",
        )
