import sys

sys.path.append("..")
from pathlib import Path

import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.plot_utils as g_plt_utils
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
import matplotlib.pyplot as plt
import numpy as np
from shell_model_experiments.params.params import *

import plot_data as sh_plt_data


def plot_energy_spectrum_comparison(args: dict):
    # if args["datapaths"] is None:
    #     raise g_exceptions.InvalidRuntimeArgument(
    #         "Argument not set", argument="datapaths"
    #     )

    axes = plt.axes()

    # Find csv files
    file_paths = g_utils.get_files_in_path(Path(args["datapath"]))
    if len(file_paths) == 0:
        raise ImportError("No files to import")

    file_paths = g_utils.sort_paths_according_to_header_dicts(
        file_paths, ["ny_n", "diff_exponent"]
    )

    for i, file_path in enumerate(file_paths):
        # Import data
        u_data, header_dict = g_import.import_data(file_path, start_line=1)

        # time, u_data, header_dict = g_import.import_ref_data(args=args)

        # Setup plot args
        plot_arg_list = ["fit_slope"]
        if i == 0:
            plot_arg_list.append("kolmogorov")

        sh_plt_data.plot_energy_spectrum(
            u_data,
            header_dict,
            axes=axes,
            plot_arg_list=plot_arg_list,
            plot_kwarg_list={"title": "Energy spectrum vs $n_{{\\nu}}$ and $\\alpha$"},
        )
        plt.legend()


def plot_helicity_spectrum_comparison(args: dict):

    axes = plt.axes()

    # Find csv files
    file_paths = g_utils.get_files_in_path(Path(args["datapath"]))

    file_paths = g_utils.sort_paths_according_to_header_dicts(
        file_paths, ["ny_n", "diff_exponent"]
    )

    for i, file_path in enumerate(file_paths):
        # Import data
        u_data, header_dict = g_import.import_data(file_path, start_line=1)

        # time, u_data, header_dict = g_import.import_ref_data(args=args)

        sh_plt_data.plot_helicity_spectrum(
            u_data,
            header_dict,
            args,
            axes=axes,
            plot_arg_list=["hel_sign"],
            plot_kwarg_list={
                "title": "Helicity spectrum vs $n_{{\\nu}}$ and $\\alpha$"
            },
        )
        axes.set_ylim(1e-6, 1e5)
        plt.legend()


def plot_period4_spectrum_ratio(args: dict):

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)
    cmap_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Find csv files
    file_paths = g_utils.get_files_in_path(Path(args["datapath"]))

    # Bounds reference (ny_n = 19) to the top index
    ref_index = [
        i for i, file_path in enumerate(file_paths) if "ny_n19.0" in file_path.name
    ][0]

    file_paths.insert(0, file_paths[ref_index])
    del file_paths[ref_index + 1]

    for i, file_path in enumerate(file_paths):
        # Import data
        u_data, header_dict = g_import.import_data(file_path, start_line=1)
        u_data = u_data.real

        if i == 0:
            ref_data = u_data

        # Continue if hyper diff is not on
        if header_dict["diff_exponent"] == 2:
            continue

        shell_limit = int(header_dict["ny_n"])
        rel_u_data = u_data / ref_data

        axes[2].plot(
            np.log2(k_vec_temp[:shell_limit]),
            rel_u_data[0, :shell_limit],
            color=cmap_list[i],
        )

        step = 3
        _slice_lower = np.s_[1:shell_limit:step]
        _slice_upper = np.s_[2:shell_limit:step]

        k_vectors_upper = np.log2(k_vec_temp[_slice_upper])
        k_vectors_lower = np.log2(k_vec_temp[_slice_lower])
        if header_dict["ny_n"] == 13:
            k_vectors_upper += step
            k_vectors_lower += step

        axes[1].plot(
            k_vectors_lower,
            rel_u_data[0, _slice_lower],
            color=cmap_list[i],
            linestyle="-",
        )

        axes[1].plot(
            k_vectors_upper,
            rel_u_data[0, _slice_upper],
            color=cmap_list[i],
            linestyle="-",
        )

        diff = rel_u_data[0, _slice_upper] - rel_u_data[0, _slice_lower]

        axes[0].plot(
            k_vectors_upper,
            diff,
            color=cmap_list[i],
            label=f"$n_{{\\nu}}$={int(header_dict['ny_n'])}, "
            + f"$\\alpha$={int(header_dict['diff_exponent'])}",
        )

    titles = []
    titles.append(
        g_plt_utils.generate_title(
            args, title_header="Peak-to-peak amplitude rel. $n_{{\\nu}}=19$"
        )
    )
    titles.append(
        g_plt_utils.generate_title(
            args, title_header="Lower and upper peaks rel. $n_{{\\nu}}=19$"
        )
    )
    titles.append(
        g_plt_utils.generate_title(args, title_header="Spectrum rel. $n_{{\\nu}}=19$")
    )

    for i, title in enumerate(titles):
        axes[i].set_title(title)

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].legend()

    plt.suptitle("Period4 hyper-diffusion investigation 1")


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()

    args = stand_plot_arg_parser.args

    g_ui.confirm_run_setup(args)

    if "energy_spec_compare" in args["plot_type"]:
        plot_energy_spectrum_comparison(args=args)

    if "helicity_spec_compare" in args["plot_type"]:
        plot_helicity_spectrum_comparison(args)

    if "period4_ratio" in args["plot_type"]:
        plot_period4_spectrum_ratio(args=args)

    g_plt_utils.save_or_show_plot(args)
