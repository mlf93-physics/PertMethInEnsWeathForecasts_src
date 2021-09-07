import sys

sys.path.append("..")
import argparse
import math
import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt
from pyinstrument import Profiler
from shell_model_experiments.params.params import *
from shell_model_experiments.utils.import_data_funcs import (
    import_perturbation_velocities,
)
import shell_model_experiments.analyses.lorentz_block_analysis as lr_analysis
import shell_model_experiments.plotting.plot_data as pl_data
import shell_model_experiments.utils.import_data_funcs as imp_funcs

profiler = Profiler()


def plt_lorentz_block_from_full_perturbation_data(args):

    if "perturb_folder" in args:

        parent_pert_folder = args["perturb_folder"]
        # Import forecasts
        args["perturb_folder"] = parent_pert_folder + "/forecasts"
        args["n_files"] = -1

        (
            forecast_pert_u_stores,
            _,
            _,
            forecast_header_dict,
        ) = import_perturbation_velocities(args)

        # Import forecasts
        args["perturb_folder"] = parent_pert_folder + "/analysis_forecasts"
        args["n_files"] = -1

        (
            ana_forecast_pert_u_stores,
            _,
            _,
            _,
        ) = import_perturbation_velocities(args)

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
                        int((day + 1) * day_offset * sample_rate / dt) + 1, :
                    ]
                    rmse_array[fc, day] = np.sqrt(
                        np.mean((_error * _error.conj()).real)
                    )
                else:
                    _error = (
                        forecast_pert_u_stores[fc][
                            int((day + 1) * day_offset * sample_rate / dt) + 1, :
                        ]
                        - ana_forecast_pert_u_stores[fc][
                            int((day - fc) * day_offset * sample_rate / dt) + 1, :
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

    legend = [f"$\\Delta = {i + 1}$" for i in range(num_forecasts)]

    # Make average plot...
    if args["average"]:
        # Get diagonal elements
        legend.append("The black")

        plt.plot(
            rmse.T,
            linewidth=1.5,
        )

        # Reset color cycle
        plt.gca().set_prop_cycle(None)

        plt.plot(rmse.T, ".", markersize=8, label="_nolegend_")

        plt.xlabel("Day")
        plt.ylabel("RMSE; $\\sqrt{\\overline{(u_{error})²}}$")
        plt.title(f"Lorentz block average | $N_{{blocks}}$={num_blocks}")
        # Plot diagonal, i.e. bold curve from [Lorentz 1982]
        plt.plot(np.diagonal(rmse), "k-")
        plt.plot(np.diagonal(rmse), "k.", markersize=8, linewidth=1.5)
        plt.legend(legend)
        # plt.yscale("log")
    # or plot each rmse in its own axes
    else:
        num_subplot_cols = math.floor(num_blocks / 2) + 1
        num_subplot_rows = math.floor(num_blocks / 2)
        fig, axes = plt.subplots(
            ncols=num_subplot_cols, nrows=num_subplot_rows, sharex=True
        )
        if len(axes.shape) == 1:
            axes = np.reshape(axes, (1, num_subplot_cols))

        for i in range(rmse.shape[-1]):
            line_plot = axes[i // num_subplot_cols, i % num_subplot_cols].plot(
                rmse[:, :, i].T
            )
            axes[i // num_subplot_cols, i % num_subplot_cols].set_xlabel("Day")
            axes[i // num_subplot_cols, i % num_subplot_cols].set_ylabel(
                "RMSE; $\\sqrt{\\overline{(u_{error})²}}$"
            )
            axes[i // num_subplot_cols, i % num_subplot_cols].set_title(
                f"Lorentz block {i+1} | $T_{{start}}$="
                + f"{ana_forecast_header_dicts[i]['perturb_pos']*dt/sample_rate}"
            )

            if i == 0:
                fig.legend(line_plot, legend, loc="center right")

        # handles, labels = axes[0, 0].get_legend_handles_labels()
        # print("handles", handles, "labels", labels)


def plt_blocks_and_energy(args):

    time, u_data, ref_header_dict = imp_funcs.import_ref_data(args=args)

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
        header_dict = imp_funcs.import_header(file_name=perturb_file_names[0])

        block_start_indices[i] = int(
            header_dict["perturb_pos"]
            + header_dict["start_time_offset"] * sample_rate / dt
        )
        block_end_indices[i] = int(
            header_dict["perturb_pos"] + header_dict["time_to_run"] * sample_rate / dt
        )
        block_names.append(block.name)

    sorted_index = np.argsort(block_start_indices)
    block_start_indices = block_start_indices[sorted_index]
    block_end_indices = block_end_indices[sorted_index]

    # Setup axes
    ax = plt.axes()
    # Calculate energy
    energy_vs_time = np.sum(u_data * np.conj(u_data), axis=1).real
    mean_energy = np.mean(energy_vs_time)

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
                int(
                    block_start_indices[i] - args["ref_start_time"] * sample_rate / dt
                ) : int(
                    block_end_indices[i] - args["ref_start_time"] * sample_rate / dt + 1
                )
            ],
            zorder=15,
            label=block_names[sorted_index[i]]
            # + 1 / num_blocks * (i + 1) * mean_energy,
        )

    # Remove perturb_folder to not plot perturbation start positions
    args["perturb_folder"] = None
    pl_data.plot_inviscid_quantities(
        time, u_data, ref_header_dict, ax=ax, omit="ny", args=args
    )
    plt.title("Lorentz block regions")
    plt.legend()


if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", nargs="?", default=None, type=str)
    arg_parser.add_argument("--perturb_folder", nargs="?", default=None, type=str)
    arg_parser.add_argument("--n_files", default=-1, type=int)
    arg_parser.add_argument("--experiment", nargs="?", default=None, type=str)
    arg_parser.add_argument("--plot_type", nargs="?", default=None, type=str)
    arg_parser.add_argument("--average", action="store_true")
    arg_parser.add_argument("--ref_start_time", default=0, type=float)
    arg_parser.add_argument("--ref_end_time", default=-1, type=float)

    args = vars(arg_parser.parse_args())

    # Add missing arguments to make util funcs work
    args["specific_ref_records"] = [0]
    args["file_offset"] = 0

    if args["plot_type"] == "blocks":
        plt_lorentz_block(args)
    elif args["plot_type"] == "blocks_and_energy":
        plt_blocks_and_energy(args)
    else:
        raise ValueError("No valid plot type given as input argument")
        exit()

    plt.tight_layout()
    plt.show()
