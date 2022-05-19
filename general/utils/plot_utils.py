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
import general.utils.importing.import_data_funcs as g_import
from general.plotting.plot_params import *
from general.params.model_licences import Models
from general.params.experiment_licences import Experiments as EXP
import config as cfg

if cfg.MODEL == Models.SHELL_MODEL:
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType

    params = PAR_SH
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params

    params = l63_params


def post_process_vectors_and_char_values(
    args: dict,
    vector_units: np.ndarray,
    characteristic_values: np.ndarray,
    header_dicts: dict,
):
    """Post process the pert vectors and characteristic values differently
    depending on the license

    E.g. the singular values are converted to growth rates and variances to fraction of
    total variance.

    Parameters
    ----------
    args : dict
        Run-time arguments
    vector_units : np.ndarray
        The vector units
    characteristic_values : np.ndarray
        The characteristic values
    header_dicts : dict
        The header dicts

    Returns
    -------
    Tuple(
        np.ndarray: valid_char_value_range
        np.ndarray: characteristic_values
    )
    """

    exp_setup = g_import.import_exp_info_file(args)

    n_zeros = 0  # Used to remove negative singular values
    if cfg.LICENCE == EXP.SINGULAR_VECTORS:
        # Cut off at zero
        real_s_values = characteristic_values.real
        below_zero_bool_array = real_s_values < 0
        # Count number of zero entries
        n_zeros = np.max(np.sum(below_zero_bool_array, axis=1))
        real_s_values[below_zero_bool_array] = 0
        temp_log_array = np.zeros(characteristic_values.shape)
        # Take log and output to temp_log_array
        np.log(
            np.sqrt(real_s_values),
            out=temp_log_array,
            where=np.logical_not(below_zero_bool_array),
        )
        # Divide by optimization time
        characteristic_values = 1 / header_dicts[0]["time_to_run"] * temp_log_array
    elif cfg.LICENCE == EXP.BREEDING_EOF_VECTORS:
        characteristic_values = (
            characteristic_values.real
            / np.sum(characteristic_values.real, axis=1)[:, np.newaxis]
        )
    else:
        characteristic_values = characteristic_values.real

    valid_char_value_range = np.s_[:-(n_zeros)] if n_zeros > 0 else np.s_[:]
    characteristic_values = characteristic_values[:, valid_char_value_range]

    sort_index = np.argsort(characteristic_values, axis=1)[:, ::-1]
    for i in range(vector_units.shape[0]):
        vector_units[i, valid_char_value_range, :] = vector_units[
            i, sort_index[i, :], :
        ]
        characteristic_values[i, :] = characteristic_values[i, sort_index[i, :]]

    return valid_char_value_range, characteristic_values


def get_non_repeating_colors(
    n_colors: int = 1,
    cmap: colors.LinearSegmentedColormap = plt.cm.gist_rainbow,
    vmin: float = 0,
    vmax: float = 1,
):
    cmap_list = [cmap(i) for i in np.linspace(vmin, vmax, n_colors)]

    return cmap_list, cmap


def get_custom_cmap(
    vmin: float = -1,
    vmax: float = 1,
    vcenter: float = 0,
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
    vcenter : float, optional
        The center value of the cmap, by default 0
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
        vcenter=vcenter,
        vmax=vmax,
    )

    return cmap, norm


def set_color_cycle_for_vectors(
    axes: plt.Axes, vector_type: str = "sv", n_vectors: int = 0
):
    if "sv" == vector_type:
        cmap_list, cmap = get_non_repeating_colors(
            n_colors=n_vectors,
            cmap=plt.cm.Blues_r,
            vmin=0.2,
            vmax=0.7,
        )
        axes.set_prop_cycle("color", cmap_list)
    elif "bv_eof" == vector_type:
        cmap_list, cmap = get_non_repeating_colors(
            n_colors=n_vectors,
            cmap=plt.cm.Oranges_r,
            vmin=0.2,
            vmax=0.7,
        )
        axes.set_prop_cycle("color", cmap_list)
    elif "bv" == vector_type:
        cmap_list, cmap = get_non_repeating_colors(
            n_colors=n_vectors,
            cmap=plt.cm.Greens_r,
            vmin=0.2,
            vmax=0.7,
        )
        axes.set_prop_cycle("color", cmap_list)
    elif "nm" == vector_type:
        cmap_list, cmap = get_non_repeating_colors(
            n_colors=n_vectors,
            cmap=plt.cm.Purples_r,
            vmin=0.2,
            vmax=0.7,
        )
        axes.set_prop_cycle("color", cmap_list)
    else:
        raise ValueError(f"No color settings for vector_type={vector_type}")

    return cmap


def save_interactive_fig(fig, subpath=None, name="standard_filename"):
    if subpath is None:
        full_path = FIG_ROOT
    else:
        full_path = FIG_ROOT / subpath

    if not os.path.isdir(full_path):
        os.makedirs(full_path)

    out_path = pl.Path(full_path, f"{name}.fig.pickle")

    with open(out_path, "wb") as file:
        pickle.dump(fig, file)


def load_interactive_fig(args):

    full_path = FIG_ROOT / args["file_path"]

    with open(full_path, "rb") as file:
        fig = pickle.load(file)
        fig.show()


def save_figure(
    args,
    subpath: pl.Path = None,
    file_name="figure1",
    fig: plt.Figure = None,
    tight_layout_rect: list = None,
):
    print("\nSaving figure...\n")
    # Prepare layout
    if not args["notight"]:
        plt.tight_layout(rect=tight_layout_rect)

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

    # Save pdf
    plot_handle.savefig(
        full_path / (file_name + ".pdf"),
        format="pdf",
        # backend="pgf",
        bbox_inches="tight",
    )

    plot_handle.savefig(
        full_path / (file_name + ".png"),
        format="png",
        dpi=100,
        bbox_inches="tight",
    )
    # plot_handle.savefig(
    #     full_path / (file_name + ".pgf"), format="pgf", bbox_inches="tight"
    # )

    print(f"\nFigures (pdf, png) saved as {file_name} at figures/{str(subpath)}\n")


def generate_title(
    args: dict,
    header_dict: dict = {},
    title_header: str = "PUT TITLE TEXT HERE",
    title_suffix: str = "",
    detailed: bool = True,
):

    exp_suffix = ""
    if "exp_folders" in args:
        if args["exp_folders"] is not None:
            exp_suffix = f'Experiments: {", ".join(args["exp_folders"])}; '
    elif "exp_folder" in args:
        if args["exp_folder"] is not None:
            exp_suffix = f'Experiment: {args["exp_folder"]}; '

    file_suffix = ""
    if "n_files" in args:
        if args["n_files"] < np.inf:
            file_suffix = f'Files: {args["file_offset"]}-{args["file_offset"] + args["n_files"]}, '

    title = ""
    if len(header_dict.keys()) > 0:
        if cfg.MODEL == Models.SHELL_MODEL:
            title = (
                f'; $\\alpha$={int(header_dict["diff_exponent"])}'
                + f', $n_{{\\nu}}$={int(header_dict["ny_n"])}, $\\nu$={header_dict["ny"]:.2e}'
                + f', time={header_dict["time_to_run"]}, '
            )
        elif cfg.MODEL == Models.LORENTZ63:
            title = (
                f'; sigma={header_dict["sigma"]}'
                + f', $b$={header_dict["b_const"]:.2e}, r={header_dict["r_const"]}'
                + f', time={header_dict["time_to_run"]}, '
            )

    model_addon = f"{str(cfg.MODEL)}, "

    # Add prefixes
    title = title_header + " | " + model_addon + title

    if detailed:
        # Add suffixes
        title += exp_suffix + file_suffix + title_suffix
    else:
        title += title_suffix

    # Strip trailing commas
    title = title.rstrip(",")
    title = title.rstrip(", ")

    # Wrap title
    title = "\n".join(
        textwrap.wrap(title, 60, break_long_words=False, break_on_hyphens=False)
    )

    return title


def save_or_show_plot(args: dict, tight_layout_rect: list = None):
    # if args["save_fig"]:
    #     subpath = pl.Path("lorentz63_experiments/singular_vectors/test_perturbations2/")

    #     for i in plt.get_fignums():
    #         fig = plt.figure(i)
    #         file_name = "sv_perturbations_sv1"

    #         name = g_ui.get_name_input(
    #             "Proposed name of figure: ", proposed_input=file_name
    #         )

    #         question = (
    #             "\nConfirm that the figure is being saved to\n"
    #             + f"path: {subpath}\n"
    #             + f"name: {name}\n"
    #         )

    #         answer = g_ui.ask_user(question)
    #         if answer:
    #             save_figure(
    #                 subpath=subpath,
    #                 file_name=name,
    #                 fig=fig,
    #                 tight_layout_rect=tight_layout_rect,
    #             )
    #         else:
    #             print("\nSaving the figure was aborted\n")

    if not args["noplot"]:
        if not args["notight"]:
            plt.tight_layout()
        plt.show()


def add_subfig_labels(axs):
    for ax, label in zip(axs, SUBFIG_LABELS):
        ax.set_title(label, loc="center", fontsize="medium")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", required=True, type=str)

    args = parser.parse_args()
    args = vars(args)

    # Load interactive figure
    load_interactive_fig(args)

    plt.show()
