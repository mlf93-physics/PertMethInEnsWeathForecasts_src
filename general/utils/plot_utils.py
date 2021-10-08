import sys

sys.path.append("..")
import argparse
import pathlib as pl
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from general.plotting.plot_params import *


def get_non_repeating_colors(n_colors=1):
    colormap = plt.cm.gist_rainbow
    cmap_list = [colormap(i) for i in np.linspace(0, 1, n_colors)]

    return cmap_list


def get_cmap_distributed_around_zero(vmin=-1, vmax=1):
    colors_neg = plt.cm.coolwarm(np.linspace(0, 0.5, 256))
    colors_pos = plt.cm.coolwarm(np.linspace(0.5, 1, 256))
    all_colors = np.vstack((colors_neg, colors_pos))
    cmap = colors.LinearSegmentedColormap.from_list("energy_map", all_colors)

    norm = colors.TwoSlopeNorm(
        vmin=vmin,
        vcenter=0,
        vmax=vmax,
    )

    return cmap, norm


def save_interactive_fig(fig, path, name):
    out_path = pl.Path(path, f"{name}.fig.pickle")

    with open(out_path, "wb") as file:
        pickle.dump(fig, file)


def load_interactive_fig(args):

    with open(args["file_path"], "rb") as file:
        fig = pickle.load(file)
        fig.show()


def save_figure(subpath: pl.Path = None, file_name="figure1"):
    print("\nSaving figure...\n")
    # Prepare layout
    plt.tight_layout()

    if subpath is None:
        full_path = FIG_ROOT
    else:
        full_path = FIG_ROOT / subpath

    # Save png
    plt.savefig(
        full_path / (file_name + ".png"),
        dpi=400,
        format="png",
    )

    # Save pgf
    plt.savefig(
        full_path / (file_name + ".pgf"),
        dpi=400,
        format="pgf",
    )

    print(f"\nFigures (png, pgf) saved as {file_name} at figures/{str(subpath)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", required=True, type=str)

    args = parser.parse_args()
    args = vars(args)

    # Load interactive figure
    load_interactive_fig(args)

    plt.show()
