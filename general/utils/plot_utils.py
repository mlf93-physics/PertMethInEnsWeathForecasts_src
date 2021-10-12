import os
import sys

sys.path.append("..")
import argparse
import pathlib as pl
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import general.utils.user_interface as g_ui
from general.plotting.plot_params import *
from general.params.model_licences import Models
from config import MODEL


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

    if not os.path.isdir(full_path):
        os.makedirs(full_path)

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


def generate_title(
    header_dict, args, title_header="PUT TITLE TEXT HERE", title_suffix=""
):

    if args["exp_folder"] is not None:
        exp_suffix = f'Experiment: {args["exp_folder"]}; '
    else:
        exp_suffix = ""

    if args["n_files"] < np.inf:
        file_suffix = (
            f'Files: {args["file_offset"]}-{args["file_offset"] + args["n_files"]} '
        )
    else:
        file_suffix = ""

    if MODEL == Models.SHELL_MODEL:
        title = (
            f'; f={header_dict["forcing"]}'
            + f', $n_{{\\nu}}$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'
            + f', time={header_dict["time_to_run"]}\n'
        )
    elif MODEL == Models.LORENTZ63:
        title = (
            f'; sigma={header_dict["sigma"]}'
            + f', $b$={header_dict["b_const"]:.2e}, r={header_dict["r_const"]}'
            + f', time={header_dict["time_to_run"]}\n'
        )

    # Add prefixes
    title = title_header + title
    # Add suffixes
    title += exp_suffix + file_suffix + title_suffix

    return title


def save_or_show_plot(args):
    if args["save_fig"]:
        subpath = pl.Path("shell_model_experiments/hyper_diffusivity/cutoff_invest")
        file_name = "hyper_diff_ny_n16_diff_exp4_cutoff1"

        question = (
            "\nConfirm that the figure is being saved to\n"
            + f"path: {subpath}\n"
            + f"name: {file_name}\n"
        )

        answer = g_ui.ask_user(question)
        if answer:
            save_figure(subpath=subpath, file_name=file_name)
        else:
            print("\nSaving the figure was aborted\n")

    elif not args["noplot"]:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", required=True, type=str)

    args = parser.parse_args()
    args = vars(args)

    # Load interactive figure
    load_interactive_fig(args)

    plt.show()
