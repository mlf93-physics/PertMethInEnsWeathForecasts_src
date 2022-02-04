import config as cfg
import general.utils.util_funcs as g_utils
import general.utils.importing.import_data_funcs as g_import
import numpy as np
from general.params.model_licences import Models
from general.utils.module_import.type_import import *

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType
    import shell_model_experiments.utils.special_params as sh_sparams

    sparams = sh_sparams
    params = PAR_SH
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams

    sparams = l63_sparams
    params = l63_params


def analyse_error_norm_vs_time(u_stores, args=None):

    if len(u_stores) == 0:
        raise IndexError("Not enough u_store arrays to compare.")

    if args["combinations"]:
        combinations = [
            [j, i] for j in range(len(u_stores)) for i in range(j + 1) if j != i
        ]
        error_norm_vs_time = np.zeros((u_stores[0].shape[0], len(combinations)))

        for enum, indices in enumerate(combinations):
            error_norm_vs_time[:, enum] = np.linalg.norm(
                u_stores[indices[0]] - u_stores[indices[1]], axis=1
            ).real
    else:
        error_norm_vs_time = np.zeros((u_stores[0].shape[0], len(u_stores)))

        for i in range(len(u_stores)):
            if len(u_stores[i]) == 1:
                u_stores[i] = np.reshape(u_stores[i], (u_stores[i].size, 1))

            error_norm_vs_time[:, i] = np.linalg.norm(u_stores[i], axis=1).real

    error_norm_mean_vs_time = np.mean(error_norm_vs_time, axis=1)

    return error_norm_vs_time, error_norm_mean_vs_time


def analyse_error_spread_vs_time_norm_of_mean(u_stores, args=None):
    """Calculates the spread of the error (u' - u) using the 'norm of the mean.'

    Formula: ||\sqrt{<((u' - u) - <u' - u>)²>}||

    """

    error_mean = np.mean(np.array(u_stores), axis=0)

    error_spread = np.array(
        [(u_stores[i] - error_mean) ** 2 for i in range(len(u_stores))]
    )

    error_spread = np.sqrt(np.mean(error_spread, axis=0))
    error_spread = np.linalg.norm(error_spread, axis=1)

    return error_spread, error_mean


def analyse_error_spread_vs_time_mean_of_norm(u_stores, args=None):
    """Calculates the spread of the error (u' - u) using the 'mean of the norm'

    Formula: \sqrt{<(||u' - u|| - <||u' - u||>)²>}

    """
    u_mean_norm = np.mean(np.linalg.norm(np.array(u_stores), axis=2).real, axis=0)

    error_spread = np.array(
        [
            (np.linalg.norm(u_stores[i], axis=1).real - u_mean_norm) ** 2
            for i in range(len(u_stores))
        ]
    )

    error_spread = np.sqrt(np.mean(error_spread, axis=0))

    return error_spread


def analyse_RMSE_and_spread_vs_time(data_array: np.ndarray, args: dict):
    """Analyse the RMSE of the ensemble mean and the spread of the ensemble members
    around the mean

    Parameters
    ----------
    data_array : np.ndarray((n_runs_per_profile, n_profiles, n_datapoints, sdim))
        The data array containing data for multiple ensembles made from same perturbation type
    args : dict
        Run-time arguments

    Returns
    -------
    [type]
        [description]
    """
    print("data_array norm", np.linalg.norm(data_array[:, 0, 0, :], axis=1))
    print("data_array", data_array.shape)
    ens_mean = np.mean(data_array, axis=0)

    print("ens_mean", ens_mean.shape)
    print("ens_mean norm", np.linalg.norm(ens_mean[0, 0, :]))

    spread_array = np.std(data_array - np.expand_dims(ens_mean, axis=0), axis=0)
    mean_spread_array = np.abs(np.mean(np.mean(spread_array, axis=2), axis=0))

    mean_RMSE_array = np.sqrt(
        np.mean(np.mean(ens_mean * ens_mean.conj(), axis=0), axis=1).real
    )

    return mean_RMSE_array, mean_spread_array


def analyse_mean_exp_growth_rate_vs_time(
    error_norm_vs_time: np.ndarray, anal_type: str = "instant", header_dicts: dict = {}
) -> np.ndarray:
    """Analyse the mean exponential growth rate vs time of one or more error norm vs
    time dataseries

    Parameters
    ----------
    error_norm_vs_time : np.ndarray(
            (number of datapoints per perturbations, number of perturbations)
        )
        An array consisting of one or more error norm vs time dataseries
    args : dict, optional
        Run-time arguments, by default None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
    (
        mean_growth_rate : The average exponential growth rate across all profiles
        profile_mean_growth_rates : The average exponential growth rate for each profile individually
    )

    """
    n_datapoints, n_perturbations = error_norm_vs_time.shape
    growth_rates = np.empty((n_datapoints - 1, n_perturbations), dtype=np.float64)

    for i in range(1, n_datapoints):
        # Instantaneous exponential growth rate
        if anal_type == "instant":
            growth_rates[i - 1, :] = (
                1
                / params.stt
                * np.log(error_norm_vs_time[i, :] / error_norm_vs_time[i - 1, :])
            )
        elif anal_type == "mean":
            # 'Averaged' (i.e. over some interval i*dt)
            growth_rates[i - 1, :] = (1 / (i * params.stt)) * np.log(
                error_norm_vs_time[i, :] / error_norm_vs_time[0, :]
            )

    # Get information about what perturbations belong to what profile
    profile_array: np.ndarray = np.array(
        g_utils.get_values_from_dicts(header_dicts, "profile"), dtype=np.int32
    )
    unique_profiles = np.unique(profile_array)
    profile_mean_growth_rates = np.empty(
        (n_datapoints - 1, unique_profiles.size), dtype=np.float64
    )

    # Average each profile individually
    for i, profile in enumerate(unique_profiles):
        profile_indices = np.argwhere(profile_array == profile).ravel()
        profile_mean_growth_rates[:, i] = np.mean(
            growth_rates[:, profile_indices], axis=1
        )

    # Average all profiles
    mean_growth_rate = np.mean(profile_mean_growth_rates, axis=1)

    return mean_growth_rate, profile_mean_growth_rates


def execute_mean_exp_growth_rate_vs_time_analysis(
    args: dict,
    u_stores: List[np.ndarray],
    header_dicts: dict = {},
    anal_type: str = "instant",
):

    # Analyse mean exponential growth rate
    (
        error_norm_vs_time,
        error_norm_mean_vs_time,
    ) = analyse_error_norm_vs_time(u_stores, args=args)
    mean_growth_rate = analyse_mean_exp_growth_rate_vs_time(
        error_norm_vs_time, anal_type=anal_type, header_dicts=header_dicts
    )

    return mean_growth_rate


def calc_eof_vectors(
    vectors: np.ndarray, n_eof_vectors: int = 2
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes an orthogonal complement to an array of vectors

    Parameters
    ----------
    vectors : np.ndarray
        The vectors to analyse. Shape: (n_units, n_vectors, sdim)

    Returns
    -------
    tuple :
        (
            np.ndarray :
                The resulting EOF vectors. Shape: (n_units, sdim, n_eof_vectors)
            np.ndarray :
                The resulting variances. Shape: (n_units, n_eof_vectors)
        )
    """
    # Get number of vectors
    n_vectors: int = vectors.shape[1]
    n_units: int = vectors.shape[0]

    # Test dimensions against n_eof_vectors requested
    if n_vectors < n_eof_vectors:
        raise ValueError(
            "The number of requested EOF vectors exceeds the number of base vectors; "
            + f"number of base vectors: {n_vectors}, number of requested EOFs: {n_eof_vectors}"
        )
    # Due to the way the breed vectors are stored, the transpose is directly given
    vectors_transpose: np.ndarray = vectors.conj()
    vectors: np.ndarray = np.transpose(vectors, axes=(0, 2, 1))

    # Calculate covariance matrix
    cov_matrix: np.ndarray = vectors_transpose @ vectors / n_vectors

    # Calculate eigenvalues and -vectors
    e_values, e_vectors = np.linalg.eig(cov_matrix)
    # Take absolute in order to avoid negative variances - observed from values
    # very close to 0 but negative - like -1e-25
    e_values = np.abs(e_values)
    sort_indices = np.argsort(e_values, axis=1)[:, ::-1]

    # Project e vectors onto the breed vectors to get the EOF vectors
    eof_vectors: np.ndarray((n_units, params.sdim, n_vectors)) = (
        vectors
        @ e_vectors
        / (
            np.sqrt(
                np.reshape(
                    e_values,
                    (n_units, 1, n_vectors),
                )
            )
        )
    )

    # Take real part if in l63 model
    if cfg.MODEL == Models.LORENTZ63:
        eof_vectors = eof_vectors.real

    # Get sorted eof vectors and e values
    eof_vectors = eof_vectors[np.arange(n_units)[:, np.newaxis], :, sort_indices]
    eof_vectors = np.transpose(eof_vectors, axes=(0, 2, 1))
    e_values = e_values[np.arange(n_units)[:, np.newaxis], sort_indices]

    # Normalize e_values
    e_values = g_utils.normalize_array(e_values, norm_value=1, axis=1)

    # Filter out unused eof's
    eof_vectors = eof_vectors[:, :, :n_eof_vectors]

    return eof_vectors, e_values
