"""Plot the number of Ki67 spots per nucleus against the nucleus volume, from the tables of compute_ki67_features.py.

Two figures: all nuclei pooled, and one panel per spheroid with shared axes. Each nucleus is a dot (the spot counts
are jittered by +-0.2 so that the integer counts do not overlap), the line is the mean count per decile of the nucleus
volume, and the title gives the Spearman correlation. The objects below MIN_VOLUME without spots are removed: they
are fragments (65 - 100% of them have no spots, above it ~35% of the nuclei).

Usage:
    python plot_ki67_spots_vs_nucleus_size.py  # writes to '<features>/plots'
"""

import os
import argparse
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

matplotlib.use("Agg")

FEATURE_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/beghin_organoids/mcf7_ki67_features"
DOT_COLOR, LINE_COLOR, TEXT_COLOR, GRID_COLOR = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
JITTER, N_BINS = 0.2, 10
# The minimal volume (um3) of the nuclei without spots, ~10.5 um equivalent diameter.
MIN_VOLUME = 600


def load_nuclei(feature_folder):
    paths = glob(os.path.join(feature_folder, "tables", "*_nuclei.csv"))
    nuclei = pd.concat(map(pd.read_csv, paths), ignore_index=True)
    nuclei["stack_number"] = nuclei["stack"].str[1:].astype(int)
    fragments = (nuclei.volume_um3 < MIN_VOLUME) & (nuclei.n_foci == 0)
    print(f"Removed {fragments.sum()} / {len(nuclei)} objects below {MIN_VOLUME} um3 without spots")
    return nuclei[~fragments].sort_values(["stack_number", "nucleus_id"], ignore_index=True)


def style_axis(ax):
    ax.grid(color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)


def plot_nuclei(ax, nuclei, rng, dot_size, n_bins=N_BINS):
    """The dots of the nuclei and the mean spot count per volume quantile, returns the Spearman correlation."""
    jitter = rng.uniform(-JITTER, JITTER, len(nuclei))
    ax.scatter(nuclei.volume_um3, nuclei.n_foci + jitter, s=dot_size, color=DOT_COLOR, alpha=0.35, linewidths=0,
               label="nucleus")
    bins = pd.qcut(nuclei.volume_um3, min(n_bins, len(nuclei) // 5), duplicates="drop")
    trend = nuclei.groupby(bins, observed=True).agg(volume=("volume_um3", "median"), n_foci=("n_foci", "mean"))
    ax.plot(trend.volume, trend.n_foci, color=LINE_COLOR, linewidth=1.5, marker="o", markersize=3,
            label=f"mean per volume {'decile' if n_bins == 10 else 'quantile'}")
    style_axis(ax)
    return spearmanr(nuclei.volume_um3, nuclei.n_foci).statistic


def plot_pooled(nuclei, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rho = plot_nuclei(ax, nuclei, np.random.default_rng(0), dot_size=6)
    ax.set_xlabel("Nucleus volume (µm³)", color=TEXT_COLOR)
    ax.set_ylabel("Ki67 spots per nucleus", color=TEXT_COLOR)
    ax.set_title(f"All spheroids: {len(nuclei)} nuclei from {nuclei['stack'].nunique()} spheroids, "
                 f"Spearman ρ = {rho:.2f}", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return rho


def plot_per_spheroid(nuclei, path, n_cols=7):
    stacks = nuclei["stack"].unique()
    n_rows = int(np.ceil(len(stacks) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 1.9 * n_rows), sharex=True, sharey=True,
                             layout="constrained")
    rng = np.random.default_rng(0)
    rhos = {}
    for ax, stack in zip(axes.flat, stacks):
        stack_nuclei = nuclei[nuclei["stack"] == stack]
        rhos[stack] = plot_nuclei(ax, stack_nuclei, rng, dot_size=4, n_bins=5)
        ax.set_title(f"{stack}: n = {len(stack_nuclei)}, ρ = {rhos[stack]:.2f}", fontsize=8, loc="left")
    for ax in axes.flat[len(stacks):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", frameon=False, fontsize=9)
    fig.supxlabel("Nucleus volume (µm³)", color=TEXT_COLOR)
    fig.supylabel("Ki67 spots per nucleus", color=TEXT_COLOR)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return rhos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--feature_folder", default=FEATURE_ROOT, help="The output of compute_ki67_features.py")
    parser.add_argument("-o", "--output_folder", help="The folder for the plots, default '<feature_folder>/plots'.")
    args = parser.parse_args()

    output_folder = args.output_folder or os.path.join(args.feature_folder, "plots")
    os.makedirs(output_folder, exist_ok=True)
    nuclei = load_nuclei(args.feature_folder)
    rho = plot_pooled(nuclei, os.path.join(output_folder, "ki67_spots_vs_nucleus_volume_pooled.png"))
    rhos = plot_per_spheroid(nuclei, os.path.join(output_folder, "ki67_spots_vs_nucleus_volume_per_spheroid.png"))
    print(f"Pooled Spearman rho {rho:.2f}, per spheroid median {np.median(list(rhos.values())):.2f} "
          f"(range {min(rhos.values()):.2f} - {max(rhos.values()):.2f}), plots in {output_folder}")


if __name__ == "__main__":
    main()
