import sys

sys.path.append("..")
import pathlib as pl

import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.exceptions as g_exceptions
import general.utils.importing.import_data_funcs as g_import
import general.utils.saving.save_data_funcs as g_save
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
import numpy as np
import scipy.stats as sp_stats
import shell_model_experiments.utils.util_funcs as ut_funcs
from general.utils.module_import.type_import import *
from shell_model_experiments.params.params import PAR, ParamsStructType

cfg.GLOBAL_PARAMS.ref_run = False


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


def analyse_mean_energy_spectrum(args: dict) -> Tuple[np.ndarray, dict]:

    _, u_data, header_dict = g_import.import_ref_data(args=args)

    g_utils.determine_params_from_header_dict(header_dict, args)

    mean_energy = get_mean_energy(u_data)

    return mean_energy, header_dict


def analyse_mean_energy_spectra(args: dict):

    if args["datapaths"] is None:
        raise g_exceptions.InvalidRuntimeArgument(
            "Argument not set", argument="datapaths"
        )

    for i, path in enumerate(args["datapaths"]):
        args["datapath"] = path

        mean_energy, _ = analyse_mean_energy_spectrum(args)

        mean_energy = np.reshape(mean_energy, (1, mean_energy.size))

        g_save.save_data(mean_energy, prefix="mean_energy_", args=args)


def analyse_mean_helicity_spectrum(args: dict) -> Tuple[np.ndarray, dict]:

    _, u_data, header_dict = g_import.import_ref_data(args=args)

    g_utils.determine_params_from_header_dict(header_dict, args)

    mean_helicity = get_mean_helicity(u_data)

    return mean_helicity, header_dict


def analyse_mean_helicity_spectra(args: dict):

    if args["datapaths"] is None:
        raise g_exceptions.InvalidRuntimeArgument(
            "Argument not set", argument="datapaths"
        )

    for _, path in enumerate(args["datapaths"]):
        args["datapath"] = path

        mean_helicity, _ = analyse_mean_helicity_spectrum(args)

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


def temp_energy_relation(args):

    # Find csv files
    file_paths = g_utils.get_files_in_path(pl.Path(args["datapath"]))

    file_paths = g_utils.sort_paths_according_to_header_dicts(
        file_paths, ["ny_n", "diff_exponent"]
    )

    for i, file_path in enumerate(file_paths):
        # Import data
        u_data, header_dict = g_import.import_data(file_path, start_line=1)
        print("u_data", u_data.shape)

        mean_energy = get_mean_energy(u_data)

        shellN = int(header_dict["ny_n"]) - 1

        theta_n_2 = np.angle(u_data[0, shellN - 2])

        En_3_factor = PAR.lambda_const ** 2 * (
            1 + (PAR.epsilon / (PAR.epsilon - 1)) ** 2
        ) + 2 * PAR.lambda_const * PAR.epsilon / (PAR.epsilon - 1) * np.cos(
            2 * theta_n_2
        )

        print("file_path", file_path.name)
        print("$E_{{N-3}}$ factor", En_3_factor)
        print(
            f"$E_{{N}}$ = {mean_energy[shellN]}",
            f"$E_{{N-3}}$ = {mean_energy[shellN-3]} (data)",
            f"$E_{{N-3}}$ = {En_3_factor*mean_energy[shellN]} (calculated)",
        )


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()

    args = stand_plot_arg_parser.args

    g_ui.confirm_run_setup(args)

    # Initiate and update variables and arrays
    ut_funcs.update_dependent_params(PAR, sdim=int(args["sdim"]))
    ut_funcs.update_arrays(PAR)

    # analyse_mean_energy_spectra(args)
    analyse_mean_helicity_spectra(args)
    # temp_energy_relation(args)
