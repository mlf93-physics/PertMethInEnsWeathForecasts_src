import sys

sys.path.append("..")
import argparse
import math
import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt
from pyinstrument import Profiler
from shell_model_experiments.params.params import *
import shell_model_experiments.analyses.lorentz_block_analysis as lr_analysis
import shell_model_experiments.plotting.plot_data as pl_data
import general.utils.import_data_funcs as g_import
import shell_model_experiments.utils.plot_utils as plt_utils

profiler = Profiler()


def plt_lorentz_block_from_full_perturbation_data(args):

    if "perturb_folder" in args:

        parent_pert_folder = args["perturb_folder"]
        # Import forecasts
        args["perturb_folder"] = parent_pert_folder + "/forecasts"
        args["n_files"] = np.inf

        (
            forecast_pert_u_stores,
            _,
            _,
            forecast_header_dict,
            _,
        ) = g_import.import_perturbation_velocities(args)

        # Import forecasts
        args["perturb_folder"] = parent_pert_folder + "/analysis_forecasts"
        args["n_files"] = np.inf

        (
            ana_forecast_pert_u_stores,
            _,
            _,
            _,
            _,
        ) = g_import.import_perturbation_velocities(args)

        num_ana_forecasts = len(ana_forecast_pert_u_stores)
        num_forecasts = len(forecast_pert_u_stores)
        rmse_array = np.zeros((num_forecasts, num_ana_forecasts), dtype=np.float64)
        day_offset = forecast_header_dict["start_time_offset"]

        for fc in range(num_forecasts):
            for day in range(fc, num_ana_forecasts):
                if day == fc:

                    # NOTE: reference velocities are subtracted on import, so
                    # this is the forecast error directly
                    _error = forecast_pert_u_stores[fc][
                        int((day + 1) * day_offset * tts) + 1, :
                    ]
                    rmse_array[fc, day] = np.sqrt(
                        np.mean((_error * _error.conj()).real)
                    )
                else:
                    _error = (
                        forecast_pert_u_stores[fc][
                            int((day + 1) * day_offset * tts) + 1, :
                        ]
                        - ana_forecast_pert_u_stores[fc][
                            int((day - fc) * day_offset * tts) + 1, :
                        ]
                    )
                    rmse_array[fc, day] = np.sqrt(
                        np.mean((_error * _error.conj()).real)
                    )

        rmse_array[np.where(rmse_array == 0)] = float("nan")
        plt.plot(rmse_array[:num_forecasts, :].T)
        plt.show()


def plt_lorentz_block(args):

    (
        rmse,
        ana_forecast_header_dicts,
    ) = lr_analysis.lorentz_block_analyser.analysis_executer(args)

    num_blocks = len(ana_forecast_header_dicts)
    num_forecasts = rmse.shape[0]

    if num_blocks == 0:
        raise ValueError(f"No blocks to plot. num_blocks = {num_blocks}")

    legend = [f"$\\Delta = {i + 1}$" for i in range(num_forecasts)]
    # Get non-repeating colorcycle
    cmap_list = plt_utils.get_non_repeating_colors(n_colors=num_forecasts)

    # Make average plot...
    if args["average"]:
        # Get diagonal elements
        legend.append("The black")
        # Set colors
        ax = plt.gca()
        ax.set_prop_cycle("color", cmap_list)

        plt.plot(
            rmse.T,
            linewidth=1.5,
        )

        # Reset color cycle
        ax.set_prop_cycle("color", cmap_list)

        plt.plot(rmse.T, ".", markersize=8, label="_nolegend_")

        plt.xlabel("Forecast day; $k$")
        plt.ylabel("RMSE; log($\\sqrt{E_{jk}²})$")
        plt.title(f"Lorentz block average | $N_{{blocks}}$={num_blocks}")
        # Plot diagonal, i.e. bold curve from [Lorentz 1982]
        plt.plot(np.diagonal(rmse), "k-")
        plt.plot(np.diagonal(rmse), "k.", markersize=8, linewidth=1.5)
        plt.legend(legend)
        # plt.yscale("log")
    # or plot each rmse in its own axes
    else:
        num_subplot_cols = math.floor(num_blocks / 2) + 1
        num_subplot_rows = math.ceil(num_blocks / num_subplot_cols)
        fig, axes = plt.subplots(
            ncols=num_subplot_cols,
            nrows=num_subplot_rows,
            sharex=True,
            sharey=args["sharey"],
        )

        # Catch if axes is one dimensional or not an array (if only one plot to make)
        try:
            if len(axes.shape) == 1:
                axes = np.reshape(axes, (1, num_subplot_cols))
        except AttributeError:
            axes = np.reshape(np.array([axes]), (1, 1))

        for i in range(rmse.shape[-1]):
            axes[i // num_subplot_cols, i % num_subplot_cols].set_prop_cycle(
                "color", cmap_list
            )
            line_plot = axes[i // num_subplot_cols, i % num_subplot_cols].plot(
                rmse[:, :, i].T
            )
            axes[i // num_subplot_cols, i % num_subplot_cols].set_xlabel(
                "Forecast day; $k$"
            )
            axes[i // num_subplot_cols, i % num_subplot_cols].set_ylabel(
                "RMSE; $\\sqrt{E_{jk}²}$"
            )
            axes[i // num_subplot_cols, i % num_subplot_cols].set_title(
                f"Lorentz block {i+1} | $T_{{start}}$="
                + f"{ana_forecast_header_dicts[i]['perturb_pos']*dt/sample_rate:.1f}"
            )
            axes[i // num_subplot_cols, i % num_subplot_cols].set_yscale("log")

            if i == 0:
                fig.legend(line_plot, legend, loc="center right")

        plt.subplots_adjust(right=0.9)


def plt_blocks_energy_regions(args):

    time, u_data, ref_header_dict = g_import.import_ref_data(args=args)

    block_dirs = lr_analysis.get_block_dirs(args)
    num_blocks = len(block_dirs)
    block_start_indices = np.zeros(num_blocks, dtype=np.int64)
    block_end_indices = np.zeros(num_blocks, dtype=np.int64)
    block_names = []

    # Import one forecast header to get perturb positions
    for i, block in enumerate(block_dirs):
        # Get perturb_folder
        args["perturb_folder"] = str(
            pl.Path(block.parents[0].name, block.name, "forecasts")
        )

        # Get perturb file names
        perturb_file_names = list(
            pl.Path(args["path"], args["perturb_folder"]).glob("*.csv")
        )
        # Import header
        header_dict = g_import.import_header(file_name=perturb_file_names[0])

        block_start_indices[i] = int(
            header_dict["perturb_pos"] + header_dict["start_time_offset"] * tts
        )
        block_end_indices[i] = int(
            header_dict["perturb_pos"] + header_dict["time_to_run"] * tts
        )
        block_names.append(block.name)

    sorted_index = np.argsort(block_start_indices)
    block_start_indices = block_start_indices[sorted_index]
    block_end_indices = block_end_indices[sorted_index]

    # Setup axes
    ax = plt.axes()
    # Calculate energy
    energy_vs_time = np.sum(u_data * np.conj(u_data), axis=1).real

    for i in range(num_blocks):
        time_array = np.linspace(
            block_start_indices[i] * dt / sample_rate,
            block_end_indices[i] * dt / sample_rate,
            int(block_end_indices[i] - block_start_indices[i]) + 1,
            endpoint=True,
        )

        ax.plot(
            time_array,
            energy_vs_time[
                int(block_start_indices[i] - args["ref_start_time"] * tts) : int(
                    block_end_indices[i] - args["ref_start_time"] * tts + 1
                )
            ],
            zorder=15,
            label=block_names[sorted_index[i]],
        )

    # Remove perturb_folder to not plot perturbation start positions
    args["perturb_folder"] = None
    pl_data.plot_inviscid_quantities(
        time, u_data, ref_header_dict, ax=ax, omit="ny", args=args
    )
    plt.title("Lorentz block regions")
    plt.legend()


def plt_block_and_energy(args):

    # Only plot one block
    args["num_blocks"] = 1

    (
        rmse,
        ana_forecast_header_dicts,
    ) = lr_analysis.lorentz_block_analyser.analysis_executer(args)

    num_forecasts = rmse.shape[0]

    block_legend = [f"$\\Delta = {i + 1}$" for i in range(num_forecasts)]
    block_legend.append("The black")

    # Setup axes
    fig, axes = plt.subplots(ncols=1, nrows=2)

    # Get non-repeating colorcycle
    cmap_list = plt_utils.get_non_repeating_colors(n_colors=num_forecasts)
    axes[1].set_prop_cycle("color", cmap_list)
    block_handles = axes[1].plot(
        rmse[:, :, 0].T,
        linewidth=1.5,
    )

    print("block_handles", len(block_handles))

    # Reset color cycle
    axes[1].set_prop_cycle("color", cmap_list)

    axes[1].plot(rmse[:, :, 0].T, ".", markersize=8, label="_nolegend_")
    axes[1].set_xlabel("Forecast day; $k$")
    axes[1].set_ylabel("RMSE; $\\sqrt{E_{jk}²}$")
    axes[1].set_title(f"Lorentz block")
    # Plot diagonal, i.e. bold curve from [Lorentz 1982]
    block_handles.extend(axes[1].plot(np.diagonal(rmse)[0], "k-"))
    axes[1].plot(np.diagonal(rmse)[0], "k.", markersize=8, linewidth=1.5)

    plt.figlegend(
        block_handles, block_legend, loc="center right", bbox_to_anchor=(1.0, 0.5)
    )

    # Get perturb positions
    block_start_index = int(
        ana_forecast_header_dicts[0]["perturb_pos"]
        + ana_forecast_header_dicts[0]["start_time_offset"] * tts
    )
    block_end_index = int(
        ana_forecast_header_dicts[0]["perturb_pos"]
        + ana_forecast_header_dicts[0]["time_to_run"] * tts
    )

    # Import only portion of ref record that fits the actual block
    args["ref_start_time"] = block_start_index * dt / sample_rate
    args["ref_end_time"] = block_end_index * dt / sample_rate
    time, u_data, ref_header_dict = g_import.import_ref_data(args=args)

    # Remove perturb_folder to not plot perturbation start positions
    args["perturb_folder"] = None
    pl_data.plot_inviscid_quantities(
        time, u_data, ref_header_dict, ax=axes[0], omit="ny", args=args
    )
    axes[0].set_title("Lorentz block energy region")
    plt.subplots_adjust(
        top=0.955, bottom=0.075, left=0.04, right=0.9, hspace=0.253, wspace=0.2
    )


if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", nargs="?", default=None, type=str)
    arg_parser.add_argument("--perturb_folder", nargs="?", default=None, type=str)
    arg_parser.add_argument("--n_files", default=-1, type=int)
    arg_parser.add_argument("--experiment", nargs="?", default=None, type=str)
    arg_parser.add_argument("--plot_type", nargs="?", default=None, type=str)
    arg_parser.add_argument("--average", action="store_true")
    arg_parser.add_argument("--sharey", action="store_true")
    arg_parser.add_argument("--ref_start_time", default=0, type=float)
    arg_parser.add_argument("--ref_end_time", default=-1, type=float)
    arg_parser.add_argument("--num_blocks", default=np.inf, type=int)

    args = vars(arg_parser.parse_args())

    # Add missing arguments to make util funcs work
    args["specific_ref_records"] = [0]
    args["file_offset"] = 0

    if args["plot_type"] == "blocks":
        plt_lorentz_block(args)
    elif args["plot_type"] == "blocks_energy_regions":
        plt_blocks_energy_regions(args)
    elif args["plot_type"] == "blocks_and_energy":
        plt_block_and_energy(args)
    else:
        raise ValueError("No valid plot type given as input argument")
        exit()

    plt.tight_layout()
    plt.show()
