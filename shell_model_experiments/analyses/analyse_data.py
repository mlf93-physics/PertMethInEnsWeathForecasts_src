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
import numpy as np
import scipy.ndimage as sp_ndi
import scipy.stats as sp_stats
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
    print("mean_eddy_turnover", mean_eddy_turnover)

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

        g_save.save_data(mean_velocity, prefix="mean_", args=args)


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

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True)

    # Import reference data
    if args["n_ref_records"] is not None:
        for record in range(args["n_ref_records"]):
            args["specific_ref_records"] = [record]
            time, u_data, header_dict = g_import.import_ref_data(args=args)

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

    # Save array
    g_save.save_data(
        out_array,
        prefix="regime_start_times_",
        header="high=0, low=1",
        fmt=f"%.{_precision}f",
        args=args,
    )

    #         axes[0].plot(time.real, total_energy)
    #         axes[0].set_title("Total energy")
    #         axes[1].plot(time.real[:-1] + 1 / 2 * PAR.stt, diff_total_energy)
    #         axes[1].set_title("Diff. total energy")
    #         axes[2].plot(
    #             time.real[:-1] + 1 / 2 * PAR.stt,
    #             eroded_bool_array,
    #         )
    #         axes[2].set_title("Erosion/dilation filtered diff. array")
    #         axes[3].plot(
    #             time.real[:-1] + 1 / 2 * PAR.stt,
    #             high_pred_regime_starts,
    #             label="High pred start",
    #         )
    #         axes[3].plot(
    #             time.real[:-1] + 1 / 2 * PAR.stt,
    #             low_pred_regime_starts,
    #             label="Low pred start",
    #         )
    #         axes[3].set_title("Detected regime start times")
    #         axes[3].legend(
    #             loc="center right",
    #             bbox_to_anchor=(1.15, 0.5),
    #         )
    # plt.tight_layout()
    # plt.show()


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

    args["ny"] = ut_funcs.ny_from_ny_n_and_forcing(
        args["forcing"], args["ny_n"], args["diff_exponent"]
    )

    # main(args)
    find_distinct_pred_regimes(args)
