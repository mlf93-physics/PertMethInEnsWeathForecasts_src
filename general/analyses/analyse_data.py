import config as cfg
import lorentz63_experiments.params.params as l63_params
import general.utils.util_funcs as g_utils
import numpy as np
from general.params.model_licences import Models
from general.utils.module_import.type_import import *
from shell_model_experiments.params.params import PAR as PAR_SH
from shell_model_experiments.params.params import ParamsStructType

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    params = PAR_SH
elif cfg.MODEL == Models.LORENTZ63:
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


def analyse_mean_exp_growth_rate_vs_time(
    error_norm_vs_time: np.ndarray, args: dict = None
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
    """
    n_datapoints, n_perturbations = error_norm_vs_time.shape
    growth_rates = np.empty((n_datapoints - 1, n_perturbations))

    for i in range(1, n_datapoints):
        # Instantaneous exponential growth rate
        # growth_rates[i - 1, :] = (1 / (params.stt)) * np.log(
        #     error_norm_vs_time[i, :] / error_norm_vs_time[i - 1, :]
        # )
        # 'Averaged' (i.e. over some interval i*dt)
        growth_rates[i - 1, :] = (1 / (i * params.stt)) * np.log(
            error_norm_vs_time[i, :] / error_norm_vs_time[0, :]
        )

    mean_growth_rate = np.mean(growth_rates, axis=1)

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
