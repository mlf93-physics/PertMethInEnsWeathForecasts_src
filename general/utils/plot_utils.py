import os
import sys

sys.path.append("..")
import argparse
import pathlib as pl
import pickle
import textwrap
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import general.utils.user_interface as g_ui
from general.plotting.plot_params import *
from general.params.model_licences import Models
from config import MODEL


def get_non_repeating_colors(n_colors: int = 1):
    colormap = plt.cm.gist_rainbow
    cmap_list = [colormap(i) for i in np.linspace(0, 1, n_colors)]

    return cmap_list


def get_cmap_distributed_around_zero(
    vmin: float = -1,
    vmax: float = 1,
    neg_thres: float = 0.5,
    pos_thres: float = 0.5,
    cmap_handle: plt.cm = plt.cm.coolwarm,
):
    """Generate a cmap and norm distributed around zero

    Parameters
    ----------
    vmin : float, optional
        The minimum value of the cmap, by default -1
    vmax : float, optional
        The maximum value of the cmap, by default 1
    neg_thres : float, optional
        The upper threshold of the negative colours, i.e. the negative colours
        are mapped to the range [0, neg_thres], by default 0.5
    pos_thres : float, optional
        The lower threshold of the positive colours, i.e. the positive colours
        are mapped to the range [0, pos_thres], by default 0.5
    cmap_handle : plt.cm, optional
        A handle to the desired cmap, by default plt.cm.coolwarm

    Returns
    -------
    [type]
        [description]
    """
    colors_neg = cmap_handle(np.linspace(0, neg_thres, 256))
    colors_pos = cmap_handle(np.linspace(pos_thres, 1, 256))
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


def save_figure(subpath: pl.Path = None, file_name="figure1", fig: plt.Figure = None):
    print("\nSaving figure...\n")
    # Prepare layout
    plt.tight_layout()

    if subpath is None:
        full_path = FIG_ROOT
    else:
        full_path = FIG_ROOT / subpath

    if not os.path.isdir(full_path):
        os.makedirs(full_path)

    if fig is None:
        plot_handle = plt
    else:
        plot_handle = fig

    # Save png
    plot_handle.savefig(full_path / (file_name + ".png"), dpi=400, format="png")

    # Save pgf
    plot_handle.savefig(
        full_path / (file_name + ".pgf"),
        dpi=400,
        format="pgf",
    )

    print(f"\nFigures (png, pgf) saved as {file_name} at figures/{str(subpath)}\n")


def generate_title(
    header_dict: dict,
    args: dict,
    title_header: str = "PUT TITLE TEXT HERE",
    title_suffix: str = "",
):

    if args["exp_folder"] is not None:
        exp_suffix = f'Experiment: {args["exp_folder"]}; '
    else:
        exp_suffix = ""

    if args["n_files"] < np.inf:
        file_suffix = (
            f'Files: {args["file_offset"]}-{args["file_offset"] + args["n_files"]}, '
        )
    else:
        file_suffix = ""

    if MODEL == Models.SHELL_MODEL:
        title = (
            f'; $\\alpha$={int(header_dict["diff_exponent"])}'
            + f', $n_{{\\nu}}$={int(header_dict["ny_n"])}, $\\nu$={header_dict["ny"]:.2e}'
            + f', time={header_dict["time_to_run"]}, '
        )
    elif MODEL == Models.LORENTZ63:
        title = (
            f'; sigma={header_dict["sigma"]}'
            + f', $b$={header_dict["b_const"]:.2e}, r={header_dict["r_const"]}'
            + f', time={header_dict["time_to_run"]}, '
        )

    # Add prefixes
    title = title_header + title
    # Add suffixes
    title += exp_suffix + file_suffix + title_suffix

    # Strip trailing commas
    title = title.rstrip(",")
    title = title.rstrip(", ")

    # Wrap title
    title = "\n".join(textwrap.wrap(title, 40))

    return title


def save_or_show_plot(args: dict):
    if args["save_fig"]:
        subpath = pl.Path(
            "shell_model_experiments/hyper_diffusivity/lyapunov_fourier_correspondenceTest"
        )

        for i in plt.get_fignums():
            fig = plt.figure(i)
            file_name = "lyapunov_fourier_correspondence_hyp_diff_comparison"

            name = g_ui.get_name_input(
                "Proposed name of figure: ", proposed_input=file_name
            )

            question = (
                "\nConfirm that the figure is being saved to\n"
                + f"path: {subpath}\n"
                + f"name: {name}\n"
            )

            answer = g_ui.ask_user(question)
            if answer:
                save_figure(subpath=subpath, file_name=name, fig=fig)
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
