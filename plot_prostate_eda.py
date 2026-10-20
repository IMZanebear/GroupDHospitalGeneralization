"""Reproduce the three six-site prostate MRI EDA figures from the metadata CSV.

Usage:
    pip install pandas numpy matplotlib
    python plot_prostate_eda.py --csv "processed_nii_metadata.csv"

Outputs are saved as PNG files in the directory specified by --output-dir.
"""
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SITE_ORDER = ["RUNMC", "BMC", "I2CVB", "UCL", "BIDMC", "HK"]
SITE_LABELS = ["Site A", "Site B", "Site C", "Site D", "Site E", "Site F"]
# Matplotlib's tab10 colors match the original screenshots.
SITE_COLORS = plt.get_cmap("tab10").colors[:6]


def scatter_by_site(ax, df, column, title, ylabel=None, panel_label=None, seed=42):
    """Plot one dot per patient, jittered horizontally to show overlapping values."""
    rng = np.random.default_rng(seed)
    for index, site in enumerate(SITE_ORDER):
        # errors="coerce": values that cannot be converted are replaced with NaN
        values = pd.to_numeric(
            df.loc[df["site"] == site, column], errors="coerce"
        ).dropna().to_numpy()
        # Horizontal jitter does not alter any measured values.
        jitter = rng.uniform(-0.18, 0.18, len(values))
        ax.scatter(index + jitter, values, s=30, alpha=0.8,
                   color=SITE_COLORS[index], edgecolors="none")

    ax.set_xticks(range(len(SITE_ORDER)), SITE_LABELS)
    ax.set_title(title, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Site")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    if panel_label:
        ax.text(-0.07, 1.04, panel_label, transform=ax.transAxes,
                fontsize=24, color="0.35", va="top", ha="right")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("processed_nii_metadata.csv"),
                        help="CSV created by extract_processed_nii_metadata.py")
    parser.add_argument("--output-dir", type=Path, default=Path("eda_figures"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    needed = {"site", "Z", "sx_mm", "sz_mm", "raw_median",
              "foreground_fraction", "prostate_volume_ml"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: geometry (two top panels and one bottom-left panel).
    fig, axs = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    scatter_by_site(axs[0, 0], df, "Z", "Image depth by site", "Slices", "A.")
    scatter_by_site(axs[0, 1], df, "sx_mm", "In-plane spacing by site",
                    "In-plane spacing (mm)", "B.")
    scatter_by_site(axs[1, 0], df, "sz_mm", "Through-plane spacing by site",
                    "Through-plane spacing (mm)")
    axs[1, 1].axis("off")
    fig.savefig(args.output_dir / "figure1_image_geometry.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: raw intensity (one full-width panel).
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    scatter_by_site(ax, df, "raw_median", "Raw median intensity by site",
                    "Raw median intensity")
    fig.savefig(args.output_dir / "figure2_raw_median_intensity.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: label occupancy and physical prostate volume.
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    scatter_by_site(axs[0], df, "foreground_fraction", "Foreground fraction by site",
                    "Foreground fraction", "A.")
    scatter_by_site(axs[1], df, "prostate_volume_ml",
                    "Physical prostate volume by site", "Prostate volume (mL)", "B.")
    fig.savefig(args.output_dir / "figure3_prostate_segmentation.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Plotted {len(df)} cases across {df['site'].nunique()} sites.")
    for path in sorted(args.output_dir.glob("figure*.png")):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
