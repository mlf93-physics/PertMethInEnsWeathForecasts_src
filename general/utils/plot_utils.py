import argparse
import pathlib as pl
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", required=True, type=str)

    args = parser.parse_args()
    args = vars(args)

    # Load interactive figure
    load_interactive_fig(args)

    plt.show()
