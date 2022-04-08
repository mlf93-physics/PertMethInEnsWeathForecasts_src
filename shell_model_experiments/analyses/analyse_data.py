import sys

sys.path.append("..")

import decimal
import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.exceptions as g_exceptions
import general.utils.importing.import_data_funcs as g_import
import general.utils.saving.save_data_funcs as g_save
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
import general.utils.plot_utils as g_plt_utils
import general.plotting.plot_config as plt_config
import numpy as np
import scipy.ndimage as sp_ndi
import scipy.stats as sp_stats
import matplotlib.pyplot as plt
import shell_model_experiments.utils.util_funcs as ut_funcs
from general.utils.module_import.type_import import *
from shell_model_experiments.params.params import PAR, ParamsStructType

cfg.GLOBAL_PARAMS.ref_run = False


def get_eddy_turnovertime(u_store):
    """Get eddy turnovertime from shell velocity data

    Parameters
    ----------
    u_store : np.ndarray
        The shell velocity data

    Returns
    -------
    np.ndarray
        The mean eddy turnover time
    """
    # Calculate mean eddy turnover time
    mean_u_norm = np.mean(np.sqrt(u_store * np.conj(u_store)).real, axis=0)
    mean_eddy_turnover = 2 * np.pi / (PAR.k_vec_temp * mean_u_norm)

    return mean_eddy_turnover


def get_mean_energy(u_data: np.ndarray) -> np.ndarray:
    """Get mean energy from velocity data

    Parameters
    ----------
    u_data : np.ndarray
        The velocity data

    Returns
    -------
    np.ndarray
        The mean energy
    """
    mean_energy = np.mean(
        (u_data * np.conj(u_data)).real,
        axis=0,
    )

    return mean_energy


def get_mean_helicity(u_data: np.ndarray) -> np.ndarray:
    """Get mean helicity from velocity data

    Parameters
    ----------
    u_data : np.ndarray
        The velocity data

    Returns
    -------
    np.ndarray
        The mean helicity
    """
    mean_energy = get_mean_energy(u_data)
    mean_helicity = PAR.hel_pre_factor * mean_energy

    return mean_helicity


def analyse_mean_velocity_spectrum(args: dict) -> Tuple[np.ndarray, dict]:

    _, u_data, header_dict = g_import.import_ref_data(args=args)

    g_utils.determine_params_from_header_dict(header_dict, args)

    mean_velocities = np.mean(u_data, axis=0)

    return mean_velocities, header_dict


def analyse_mean_velocity_spectra(args: dict):

    if args["datapaths"] is None:
        raise g_exceptions.InvalidRuntimeArgument(
            "Argument not set", argument="datapaths"
        )

    for i, path in enumerate(args["datapaths"]):
        args["datapath"] = path

        mean_velocity, _ = analyse_mean_velocity_spectrum(args)

        mean_velocity = np.reshape(mean_velocity, (1, mean_velocity.size))

        g_save.save_data(
            mean_velocity, prefix=f"mean_anal_time{args['ref_end_time']}_", args=args
        )


def analyse_mean_energy_spectrum(args: dict, u_data: np.ndarray, header_dict: dict):

    g_utils.determine_params_from_header_dict(header_dict, args)

    mean_energy = get_mean_energy(u_data)

    mean_energy = np.reshape(mean_energy, (1, mean_energy.size))

    g_save.save_data(mean_energy, prefix="mean_energy_", args=args)


def analyse_mean_helicity_spectrum(args: dict, u_data: np.ndarray, header_dict: dict):

    g_utils.determine_params_from_header_dict(header_dict, args)

    mean_helicity = get_mean_helicity(u_data)

    mean_helicity = np.reshape(mean_helicity, (1, mean_helicity.size))

    g_save.save_data(mean_helicity, prefix="mean_helicity_", args=args)


def fit_spectrum_slope(
    mean_u_data: np.ndarray, header_dict: dict
) -> Tuple[float, float]:
    """Fit a linear slope to a log-lin mean u data series

    Parameters
    ----------
    mean_u_data : np.ndarray
        The time-averaged u_data
    args : dict
        Run-time arguments
    header_dict : dict
        The header dict of the data file

    Returns
    -------
    Tuple[float, float]
        (
            slope : The slope of the linear fit
            intercept : The intercept of the 2. axis
        )
    """
    # Take only up to ny_n
    shell_limit = int(header_dict["ny_n"])
    mean_u_data = mean_u_data[:(shell_limit)]
    k_vectors = np.log2(PAR.k_vec_temp[:(shell_limit)])

    # Prepare data
    logged_mean_u_data = np.log(mean_u_data.real.ravel())

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(
        k_vectors, logged_mean_u_data
    )

    return slope, intercept


def main(args):
    if args["datapaths"] is None:
        raise g_exceptions.InvalidRuntimeArgument(
            "Argument not set", argument="datapaths"
        )

    for _, path in enumerate(args["datapaths"]):
        args["datapath"] = path

        _, u_data, header_dict = g_import.import_ref_data(args=args)

        args["out_exp_folder"] = "../../analysed_data/mean_energy_analysed_30shells"
        analyse_mean_energy_spectrum(args, u_data, header_dict)
        args["out_exp_folder"] = "../../analysed_data/mean_helicity_analysed_30shells"
        analyse_mean_helicity_spectrum(args, u_data, header_dict)


def find_distinct_pred_regimes(args):
    out_array = None

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)

    # Import reference data
    if args["n_ref_records"] is not None:
        for record in range(args["n_ref_records"]):
            args["specific_ref_records"] = [record]
            time, u_data, header_dict = g_import.import_ref_data(args=args)

            (
                total_energy,
                bool_diff_array,
                eroded_bool_array,
                high_pred_regime_starts_times,
                low_pred_regime_starts_times,
            ) = detect_regions(time, u_data)

            prepare_out_array(
                out_array, high_pred_regime_starts_times, low_pred_regime_starts_times
            )

            # Save array
            # g_save.save_data(
            #     out_array,
            #     prefix="regime_start_times_",
            #     header="high=0, low=1",
            #     fmt=f"%.{_precision}f",
            #     args=args,
            # )

    else:
        time, u_data, header_dict = g_import.import_ref_data(args=args)

        (
            total_energy,
            bool_diff_array,
            eroded_bool_array,
            high_pred_regime_starts_times,
            low_pred_regime_starts_times,
        ) = detect_regions(time, u_data)

        prepare_out_array(
            out_array, high_pred_regime_starts_times, low_pred_regime_starts_times
        )

        axes[0].plot(time.real, total_energy, "k")
        axes[0].set_title("Total energy")
        axes[0].set_ylabel("Energy\n$\\frac{1}{2} u_{n, ref} u_{n, ref}^*$")

        # bool_diff_array = bool_diff_array.astype(np.int8)
        # bool_diff_array[bool_diff_array == 0] = -1
        axes[1].plot(
            time.real[:-1] + 1 / 2 * PAR.stt,
            bool_diff_array,
            "k",
            linewidth=0.5,
        )
        axes[1].set_title("Bool diff. array")
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["Negative\nslope", "Positive\nslope"])

        dv = 0.02

        n_start_times = len(high_pred_regime_starts_times)
        for i in range(n_start_times):
            # Plot horizontal lines
            axes[2].plot(
                [high_pred_regime_starts_times[i], low_pred_regime_starts_times[i]],
                [1, 1],
                "k",
            )
            axes[2].plot(
                [
                    low_pred_regime_starts_times[i],
                    high_pred_regime_starts_times[i + 1]
                    if i + 1 < n_start_times
                    else time.real[-1],
                ],
                [0, 0],
                "k",
            )

            # Add vertical start/end lines
            axes[2].plot(
                [high_pred_regime_starts_times[i], high_pred_regime_starts_times[i]],
                [1 - dv, 1 + dv],
                color="k",
            )
            axes[2].plot(
                [low_pred_regime_starts_times[i], low_pred_regime_starts_times[i]],
                [1 - dv, 1 + dv],
                color="k",
            )
            axes[2].plot(
                [low_pred_regime_starts_times[i], low_pred_regime_starts_times[i]],
                [-dv, dv],
                color="k",
            )
            axes[2].plot(
                [
                    high_pred_regime_starts_times[i + 1]
                    if i + 1 < n_start_times
                    else time.real[-1],
                    high_pred_regime_starts_times[i + 1]
                    if i + 1 < n_start_times
                    else time.real[-1],
                ],
                [-dv, dv],
                color="k",
            )
        # axes[2].plot(time.real[:-1] + 1 / 2 * PAR.stt, eroded_bool_array, "k")
        axes[2].set_title("Erosion/dilation filtered diff. array")
        axes[2].set_xlabel("$t$")
        axes[2].set_yticks([0, 1])
        axes[2].set_yticklabels(["Small\nscales", "Large\nscales"])

        fig.subplots_adjust(
            top=0.928, bottom=0.131, left=0.126, right=0.992, hspace=0.5, wspace=0.2
        )

        if args["tolatex"]:
            plt_config.adjust_axes(axes)
            g_plt_utils.add_subfig_labels(axes)

        if args["save_fig"]:
            g_plt_utils.save_figure(
                args,
                subpath="thesis_figures/appendices/region_analysis_shell_model/",
                file_name="region_analysis_shell_model",
            )

        g_plt_utils.save_or_show_plot(args)


def prepare_out_array(
    out_array, high_pred_regime_starts_times, low_pred_regime_starts_times
):
    # Prepare array to be saved
    temp_out_array = np.stack(
        [high_pred_regime_starts_times, low_pred_regime_starts_times], axis=1
    )

    # Determine precision of time
    _precision = abs(decimal.Decimal(str(PAR.stt)).as_tuple().exponent)
    # Round off start times
    temp_out_array = np.round(temp_out_array, decimals=_precision)

    if out_array is None:
        out_array = temp_out_array
    else:
        out_array = np.append(out_array, temp_out_array, axis=0)


def detect_regions(time, u_data):
    # Get total energy
    total_energy = np.sum((u_data * np.conj(u_data)).real, axis=1)

    # Differentiate total energy
    diff_total_energy = (total_energy[1:] - total_energy[:-1]) / PAR.stt

    # Find positive and negative slopes
    bool_diff_array = diff_total_energy > 0

    # Prepare for erosion and dilation
    structure = np.ones(int(PAR.tts * 0.05))

    # Perform erosion and dilation to remove smallest regions in boolean array
    eroded_bool_array = sp_ndi.binary_erosion(
        bool_diff_array, origin=10, structure=structure
    )
    dilated_bool_array = sp_ndi.binary_dilation(
        eroded_bool_array, origin=10, structure=structure, iterations=2
    )
    eroded_bool_array: np.ndarray = sp_ndi.binary_erosion(
        dilated_bool_array, origin=10, structure=structure
    )

    # Find indices when the eroded boolean array changes bool value, which
    # indicates when a high or low pred regime begins.
    roll_array = np.roll(eroded_bool_array, 1)
    high_pred_regime_starts = np.logical_and(
        np.logical_not(roll_array), eroded_bool_array
    )
    low_pred_regime_starts = np.logical_and(
        roll_array, np.logical_not(eroded_bool_array)
    )

    # Convert indices to start times
    high_pred_regime_starts_times = time.real[np.where(high_pred_regime_starts)]
    low_pred_regime_starts_times = time.real[np.where(low_pred_regime_starts)]

    # Check if sizes are equal
    if high_pred_regime_starts_times.size != low_pred_regime_starts_times.size:
        raise ValueError(
            "Size of high pred and low pred regime start time arrays are not equal"
        )

    return (
        total_energy,
        bool_diff_array,
        eroded_bool_array,
        high_pred_regime_starts_times,
        low_pred_regime_starts_times,
    )


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()

    args = stand_plot_arg_parser.args

    g_ui.confirm_run_setup(args)

    # Initiate and update variables and arrays
    ut_funcs.update_dependent_params(PAR)
    ut_funcs.set_params(PAR, parameter="sdim", value=args["sdim"])
    ut_funcs.update_arrays(PAR)

    plt_config.adjust_default_fig_axes_settings(args)

    args["ny"] = ut_funcs.ny_from_ny_n_and_forcing(
        args["forcing"], args["ny_n"], args["diff_exponent"]
    )

    # main(args)
    find_distinct_pred_regimes(args)
    # analyse_mean_velocity_spectra(args)
