import sys

sys.path.append("..")
import argparse
import math
import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt
from pyinstrument import Profiler
import general.plotting.plot_config as g_plt_config
from shell_model_experiments.params.params import ParamsStructType
from shell_model_experiments.params.params import PAR as PAR_SH
import lorentz63_experiments.params.params as l63_params
import shell_model_experiments.plotting.plot_data as pl_data
import general.analyses.lorentz_block_analysis as lr_analysis
import general.utils.importing.import_data_funcs as g_import
import general.utils.plot_utils as g_plt_utils
import general.plotting.plot_params as plt_params
import general.utils.argument_parsers as a_parsers
import general.utils.exceptions as g_exceptions
from general.params.model_licences import Models
import config as cfg

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    params = PAR_SH
elif cfg.MODEL == Models.LORENTZ63:
    params = l63_params

# Setup plotting defaults
g_plt_config.setup_plotting_defaults()

profiler = Profiler()


def plt_lorentz_block_from_full_perturbation_data(args):

    if "exp_folder" in args:

        parent_pert_folder = args["exp_folder"]
        # Import forecasts
        args["exp_folder"] = parent_pert_folder + "/forecasts"
        args["n_files"] = np.inf

        (
            forecast_pert_u_stores,
            _,
            _,
            forecast_header_dicts,
            _,
        ) = g_import.import_perturbation_velocities(args)

        # Import forecasts
        args["exp_folder"] = parent_pert_folder + "/analysis_forecasts"
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
        day_offset = forecast_header_dicts[0]["start_time_offset"]

        for fc in range(num_forecasts):
            for day in range(fc, num_ana_forecasts):
                if day == fc:

                    # NOTE: reference velocities are subtracted on import, so
                    # this is the forecast error directly
                    _error = forecast_pert_u_stores[fc][
                        int((day + 1) * day_offset * params.tts) + 1, :
                    ]
                    rmse_array[fc, day] = np.sqrt(
                        np.mean((_error * _error.conj()).real)
                    )
                else:
                    _error = (
                        forecast_pert_u_stores[fc][
                            int((day + 1) * day_offset * params.tts) + 1, :
                        ]
                        - ana_forecast_pert_u_stores[fc][
                            int((day - fc) * day_offset * params.tts) + 1, :
                        ]
                    )
                    rmse_array[fc, day] = np.sqrt(
                        np.mean((_error * _error.conj()).real)
                    )

        rmse_array[np.where(rmse_array == 0)] = float("nan")
        plt.plot(rmse_array[:num_forecasts, :].T)
        plt.show()


def plt_lorentz_block(args):

    if args["exp_folders"] is None:
        experiments = [args["exp_folder"]]
    else:
        experiments = args["exp_folders"]

    legend = []

    for j, experiment in enumerate(experiments):
        args["exp_folder"] = experiment

        # Get experiment info
        exp_info = g_import.import_exp_info_file(args)
        # Prepare time series
        time_array = np.array(
            [exp_info["day_offset"] * (i + 1) for i in range(exp_info["n_analyses"])]
        )

        # Get Lorentz rmse block
        (
            rmse,
            ana_forecast_header_dicts,
        ) = lr_analysis.lorentz_block_analyser.analysis_executer(args)

        n_units = len(ana_forecast_header_dicts)
        num_forecasts = rmse.shape[0]

        if n_units == 0:
            raise ValueError(f"No blocks to plot. n_units = {n_units}")

        # Prepare legend
        # if len(experiments) > 1:
        legend.append(experiment + f" | $N_{{blocks}}$={n_units}")
        # else:
        #     legend = [f"$\\Delta_{{{i + 1}}}$" for i in range(num_forecasts)]
        # Get non-repeating colorcycle
        cmap_list = g_plt_utils.get_non_repeating_colors(n_colors=num_forecasts)

        # Make average plot...
        if args["average"]:
            legend.append(f"$\\Delta_{{0-{num_forecasts}}}$")
            # Set colors
            ax = plt.gca()
            # ax.set_prop_cycle("color", cmap_list)

            for k in range(num_forecasts):
                plt.plot(
                    time_array,
                    rmse.T[:, k],
                    "b",
                    alpha=0.6,
                    linewidth=1.5,
                    linestyle=plt_params.LINESTYLES[j],
                    label="_nolegend"
                    if k > 0
                    else legend[int(len(legend) // len(experiments) * j)],
                )

            # Reset color cycle
            # ax.set_prop_cycle("color", cmap_list)

            # Plot diagonal elements
            plt.plot(
                time_array,
                rmse.T,
                "b.",
                alpha=0.6,
                markersize=8,
                label="_nolegend_",
            )

            plt.xlabel("Forecast time")
            plt.ylabel("RMSE; log($\\sqrt{E_{jk}²})$")
            plt.title(f"Lorentz block average")
            # Plot diagonal, i.e. bold curve from [Lorentz 1982]
            plt.plot(
                time_array,
                np.diagonal(rmse),
                "k",
                linestyle=plt_params.LINESTYLES[j],
                label=legend[int(len(legend) // len(experiments) * j + 1)],
            )
            plt.plot(
                time_array,
                np.diagonal(rmse),
                "k.",
                markersize=8,
                linewidth=1.5,
            )
            plt.legend()
        # or plot each rmse in its own axes
        else:
            num_subplot_cols = math.floor(n_units / 2) + 1
            num_subplot_rows = math.ceil(n_units / num_subplot_cols)
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
                    time_array, rmse[:, :, i].T
                )
                axes[i // num_subplot_cols, i % num_subplot_cols].set_xlabel(
                    "Forecast time"
                )
                axes[i // num_subplot_cols, i % num_subplot_cols].set_ylabel(
                    "RMSE; $\\sqrt{E_{jk}²}$"
                )
                axes[i // num_subplot_cols, i % num_subplot_cols].set_title(
                    f"Lorentz block {i+1} | $T_{{start}}$="
                    + f"{ana_forecast_header_dicts[i]['perturb_pos']*params.stt:.1f}"
                )
                axes[i // num_subplot_cols, i % num_subplot_cols].set_yscale("log")

                if i == 0:
                    fig.legend(line_plot, legend, loc="center right")

            plt.subplots_adjust(right=0.9)


def plt_blocks_energy_regions(args):

    time, u_data, ref_header_dict = g_import.import_ref_data(args=args)

    block_dirs = lr_analysis.get_block_dirs(args)
    n_units = len(block_dirs)
    block_start_indices = np.zeros(n_units, dtype=np.int64)
    block_end_indices = np.zeros(n_units, dtype=np.int64)
    block_names = []

    # Import one forecast header to get perturb positions
    for i, block in enumerate(block_dirs):
        # Get perturb_folder
        args["exp_folder"] = str(
            pl.Path(block.parents[0].name, block.name, "forecasts")
        )

        # Get perturb file names
        perturb_file_names = list(
            pl.Path(args["datapath"], args["exp_folder"]).glob("*.csv")
        )
        # Import header
        header_dict = g_import.import_header(file_name=perturb_file_names[0])

        block_start_indices[i] = int(
            header_dict["perturb_pos"] + header_dict["start_time_offset"] * params.tts
        )
        block_end_indices[i] = int(
            header_dict["perturb_pos"] + header_dict["time_to_run"] * params.tts
        )
        block_names.append(block.name)

    sorted_index = np.argsort(block_start_indices)
    block_start_indices = block_start_indices[sorted_index]
    block_end_indices = block_end_indices[sorted_index]

    # Setup axes
    ax = plt.axes()
    # Calculate energy
    energy_vs_time = np.sum(u_data * np.conj(u_data), axis=1).real

    for i in range(n_units):
        time_array = np.linspace(
            block_start_indices[i] * params.stt,
            block_end_indices[i] * params.stt,
            int(block_end_indices[i] - block_start_indices[i]) + 1,
            endpoint=True,
        )

        ax.plot(
            time_array,
            energy_vs_time[
                int(block_start_indices[i] - args["ref_start_time"] * params.tts) : int(
                    block_end_indices[i] - args["ref_start_time"] * params.tts + 1
                )
            ],
            zorder=15,
            label=block_names[sorted_index[i]],
        )

    # Remove perturb_folder to not plot perturbation start positions
    args["exp_folder"] = None
    pl_data.plot_inviscid_quantities(
        time, u_data, ref_header_dict, ax=ax, omit="ny", args=args
    )
    plt.title("Lorentz block regions")
    plt.legend()


def plt_block_and_energy(args):

    # Only plot one block
    args["n_units"] = 1

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
    cmap_list = g_plt_utils.get_non_repeating_colors(n_colors=num_forecasts)
    axes[1].set_prop_cycle("color", cmap_list)
    block_handles = axes[1].plot(
        rmse[:, :, 0].T,
        linewidth=1.5,
    )

    # Reset color cycle
    axes[1].set_prop_cycle("color", cmap_list)

    axes[1].plot(rmse[:, :, 0].T, ".", markersize=8, label="_nolegend_")
    axes[1].set_xlabel("Forecast time")
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
        + ana_forecast_header_dicts[0]["start_time_offset"] * params.tts
    )
    block_end_index = int(
        ana_forecast_header_dicts[0]["perturb_pos"]
        + ana_forecast_header_dicts[0]["time_to_run"] * params.tts
    )

    # Import only portion of ref record that fits the actual block
    args["ref_start_time"] = block_start_index * params.stt
    args["ref_end_time"] = block_end_index * params.stt
    time, u_data, ref_header_dict = g_import.import_ref_data(args=args)

    # Remove perturb_folder to not plot perturbation start positions
    args["exp_folder"] = None
    pl_data.plot_inviscid_quantities(
        time, u_data, ref_header_dict, ax=axes[0], omit="ny", args=args
    )
    axes[0].set_title("Lorentz block energy region")
    plt.subplots_adjust(
        top=0.955, bottom=0.075, left=0.04, right=0.9, hspace=0.253, wspace=0.2
    )


if __name__ == "__main__":
    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    compare_plt_arg_parser = a_parsers.ComparisonPlottingArgParser()
    compare_plt_arg_parser.setup_parser()
    args = compare_plt_arg_parser.args

    # Check arguments
    if args["exp_folder"] is None:
        raise g_exceptions.InvalidRuntimeArgument(
            "Argument is None -" + " please select an experiment folder",
            argument="exp_folder",
        )

    if "blocks" in args["plot_type"]:
        plt_lorentz_block(args)
    elif "blocks_energy_regions" in args["plot_type"]:
        plt_blocks_energy_regions(args)
    elif "blocks_and_energy" in args["plot_type"]:
        plt_block_and_energy(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    if not args["noplot"]:
        plt.tight_layout()
        plt.show()
