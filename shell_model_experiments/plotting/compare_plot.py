import sys

sys.path.append("..")
import pathlib as pl

import config as cfg
from general.utils.module_import.type_import import *
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.plot_utils as g_plt_utils
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
from general.plotting.plot_params import *
import shell_model_experiments.analyses.analyse_data as sh_analysis
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
    file_paths = g_utils.get_files_in_path(pl.Path(args["datapath"]))
    if len(file_paths) == 0:
        raise ImportError("No files to import")

    file_paths = g_utils.sort_paths_according_to_header_dicts(
        file_paths, ["ny_n", "diff_exponent"]
    )

    header_dicts = g_utils.get_header_dicts_from_paths(file_paths)

    # Get the unique values for ny_n and alpha
    ny_n_values = g_utils.get_values_from_dicts(header_dicts, "ny_n")
    ny_n_values = sorted(list(set(ny_n_values)))

    ny_n_counter_array = np.zeros(len(ny_n_values))

    # Get colors
    cmap_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for _, file_path in enumerate(file_paths):
        # Import data
        u_data, header_dict = g_import.import_data(file_path, start_line=1)

        # time, u_data, header_dict = g_import.import_ref_data(args=args)

        # Setup plot args
        plot_arg_list = ["rel_fit"]
        # if i == 0:
        #     plot_arg_list.append("kolmogorov")

        # Get ny_n index
        ny_n_index = ny_n_values.index(int(header_dict["ny_n"]))

        # Prepare label
        label = "_nolegend_"
        if ny_n_counter_array[ny_n_index] == 0:
            label = f"$n_{{\\nu}}$={int(header_dict['ny_n'])}"

        plot_kwargs = {
            "title": "Energy spectrum vs $n_{{\\nu}}$ and $\\alpha$ rel. fit",
            "color": cmap_list[ny_n_index],
            "label": label,
        }

        sh_plt_data.plot_energy_spectrum(
            u_data,
            header_dict,
            axes=axes,
            plot_arg_list=plot_arg_list,
            plot_kwargs=plot_kwargs,
            args=args,
        )
        ny_n_counter_array[ny_n_index] += 1
    plt.legend()


def plot_helicity_spectrum_comparison(args: dict):

    axes = plt.axes()

    # Find csv files
    file_paths = g_utils.get_files_in_path(pl.Path(args["datapath"]))

    file_paths = g_utils.sort_paths_according_to_header_dicts(
        file_paths, ["ny_n", "diff_exponent"]
    )

    header_dicts = g_utils.get_header_dicts_from_paths(file_paths)

    # Get the unique values for ny_n and alpha
    ny_n_values = g_utils.get_values_from_dicts(header_dicts, "ny_n")
    ny_n_values = sorted(list(set(ny_n_values)))

    ny_n_counter_array = np.zeros(len(ny_n_values))

    # Get colors
    cmap_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, file_path in enumerate(file_paths):
        # Import data
        u_data, header_dict = g_import.import_data(file_path, start_line=1)

        plot_arg_list = [
            "hel_sign",
        ]
        if i == 0:
            plot_arg_list.append("kolmogorov")

        # Get ny_n index
        ny_n_index = ny_n_values.index(int(header_dict["ny_n"]))

        # Prepare label
        label = "_nolegend_"
        if ny_n_counter_array[ny_n_index] == 0:
            label = f"$n_{{\\nu}}$={int(header_dict['ny_n'])}"

        plot_kwargs = {
            "title": "Helicity spectrum vs $n_{{\\nu}}$ and $\\alpha$",
            "color": cmap_list[ny_n_index],
            "label": label,
        }

        sh_plt_data.plot_helicity_spectrum(
            u_data,
            header_dict,
            args,
            axes=axes,
            plot_arg_list=plot_arg_list,
            plot_kwargs=plot_kwargs,
        )
        axes.set_ylim(1e-3, 1e5)
        ny_n_counter_array[ny_n_index] += 1
        plt.legend()


def plot_period4_spectrum_ratio(args: dict):

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)

    # Find csv files
    file_paths = g_utils.get_files_in_path(pl.Path(args["datapath"]))

    # Bounds reference (ny_n = 19) to the top index
    ref_index = [
        i for i, file_path in enumerate(file_paths) if "ny_n19.0" in file_path.name
    ][0]

    file_paths.insert(0, file_paths[ref_index])
    del file_paths[ref_index + 1]

    # Get colors
    cmap_list = g_plt_utils.get_non_repeating_colors(n_colors=len(file_paths))

    for i, file_path in enumerate(file_paths):
        # Import data
        u_data, header_dict = g_import.import_data(file_path, start_line=1)
        u_data = u_data.real

        # if i == 0:
        #     ref_data = u_data
        # Fit data
        slope, intercept = sh_analysis.fit_spectrum_slope(u_data, header_dict)
        # Set ref data from slope and intercept
        ref_data = np.exp(slope * np.log2(k_vec_temp) + intercept)

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
        g_plt_utils.generate_title(args, title_header="Peak-to-peak amplitude rel. fit")
    )
    titles.append(
        g_plt_utils.generate_title(args, title_header="Lower and upper peaks rel. fit")
    )
    titles.append(g_plt_utils.generate_title(args, title_header="Spectrum rel. fit"))

    for i, title in enumerate(titles):
        axes[i].set_title(title)

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[0].legend()

    plt.suptitle("Period4 hyper-diffusion investigation 1")


def plot_period4_spectrum_ratio_vs_alpha(args: dict):

    fig, axes = plt.subplots(nrows=1, ncols=1)

    # Find csv files
    file_paths = g_utils.get_files_in_path(pl.Path(args["datapath"]))
    # Sort files
    file_paths = g_utils.sort_paths_according_to_header_dicts(
        file_paths, ["ny_n", "diff_exponent"]
    )

    header_dicts = g_utils.get_header_dicts_from_paths(file_paths)

    # Get the unique values for ny_n and alpha
    ny_n_values = g_utils.get_values_from_dicts(header_dicts, "ny_n")
    ny_n_values = list(set(ny_n_values))
    alpha_values = g_utils.get_values_from_dicts(header_dicts, "diff_exponent")
    alpha_values = sorted(list(set(alpha_values)))

    # Bounds reference (ny_n = 19) to the top index
    ref_index = [
        i for i, file_path in enumerate(file_paths) if "ny_n19.0" in file_path.name
    ][0]
    file_paths.insert(0, file_paths[ref_index])
    del file_paths[ref_index + 1]

    # Get colors
    # cmap_list = g_plt_utils.get_non_repeating_colors(n_colors=len(file_paths))
    cmap_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Prepare diff data collection
    diff_data_collection = {}
    # Define length of one period
    step = 3

    for i, file_path in enumerate(file_paths):
        # Import data
        u_data, header_dict = g_import.import_data(file_path, start_line=1)
        u_data = u_data.real

        if header_dict["ny_n"] == 19:
            continue

        # Save ref data since it's at the first index due to above trick
        # if i == 0:
        #     ref_data = u_data

        # Fit data
        slope, intercept = sh_analysis.fit_spectrum_slope(u_data, header_dict)
        # Set ref data from slope and intercept
        ref_data = np.exp(slope * np.log2(k_vec_temp) + intercept)

        # Continue if hyper diff is not on
        if header_dict["diff_exponent"] == 2:
            continue

        # Initiate list if not present
        current_ny_n = int(header_dict["ny_n"])
        if current_ny_n not in diff_data_collection:
            diff_data_collection[current_ny_n] = []

        # Normalize to reference data
        rel_u_data = u_data / ref_data
        # Prepare index slices to get upper and lower peak of period
        _slice_lower = np.s_[1 : current_ny_n - 1 : step]
        _slice_upper = np.s_[2 : current_ny_n - 1 : step]

        # calculate difference data between upper and lower peaks
        diff = np.log(rel_u_data[0, _slice_upper]) - np.log(rel_u_data[0, _slice_lower])
        diff_data_collection[current_ny_n].append(diff)

    for key in diff_data_collection.keys():
        # Convert to array
        diff_data = np.array(diff_data_collection[key])

        lines = axes.plot(
            alpha_values[1:],
            diff_data,
            color=cmap_list[ny_n_values.index(key)],
        )

        lines[0].set_label(f"$n_{{\\nu}}$={int(key)}")

    # Annotate last curves with period count
    for i in range(diff_data.shape[1]):
        axes.annotate(
            xy=(alpha_values[-1], diff_data[-1, i]),
            xytext=(5, 0),
            textcoords="offset points",
            text=f"Period #{i+1}",
            va="center",
        )

    title = g_plt_utils.generate_title(
        args, title_header="Peak-to-peak amplitude rel. fit vs. $\\alpha$"
    )
    # axes.set_yscale("log")
    axes.set_xlabel("$\\alpha$")
    axes.set_ylabel("Peak-to-peak amplitude rel. fit")
    axes.set_title(title)
    axes.legend()


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

    if "period4_ratio_vs_alpha" in args["plot_type"]:
        plot_period4_spectrum_ratio_vs_alpha(args=args)

    g_plt_utils.save_or_show_plot(args)
