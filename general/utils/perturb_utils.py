import sys

sys.path.append("..")
from math import floor, log10
import numpy as np
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
import general.utils.dev_plots as g_dev_plots
from general.params.model_licences import Models
from config import MODEL

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params


def calculate_perturbations(perturb_e_vectors, dev_plot_active=False, args=None):
    """Calculate a random perturbation with a specific norm for each profile.

    The norm of the error is defined in the parameter seeked_error_norm

    Parameters
    ----------
    perturb_e_vectors : ndarray
        The eigenvectors along which to perform the perturbations

    Returns
    -------
    perturbations : ndarray
        The random perturbations

    """
    n_profiles = args["n_profiles"]
    n_runs_per_profile = args["n_runs_per_profile"]
    perturbations = np.zeros(
        (params.sdim + 2 * params.bd_size, n_profiles * n_runs_per_profile),
        dtype=params.dtype,
    )

    if args["eigen_perturb"]:
        # Get complex-conjugate vector pair
        perturb_e_vectors_conj = np.conj(perturb_e_vectors)

    # Perform perturbation for all eigenvectors
    for i in range(n_profiles * n_runs_per_profile):
        # Apply single shell perturbation
        if args["single_shell_perturb"] is not None:
            perturb = np.zeros(params.sdim, dtype=params.dtype)
            perturb.real[args["single_shell_perturb"]] = (
                np.random.rand(1)[0].astype(np.float64) * 2 - 1
            )
            perturb.imag[args["single_shell_perturb"]] = (
                np.random.rand(1)[0].astype(np.float64) * 2 - 1
            )
        elif not args["eigen_perturb"]:
            # Generate random perturbation error
            # Reshape into complex array
            perturb = np.empty(params.sdim, dtype=params.dtype)
            # Generate random error
            error = np.random.rand(2 * params.sdim).astype(np.float64) * 2 - 1

            if MODEL == Models.SHELL_MODEL:
                perturb.real = error[: params.sdim]
                perturb.imag = error[params.sdim :]
            elif MODEL == Models.LORENTZ63:
                perturb = error[: params.sdim]

        elif args["eigen_perturb"]:
            # Generate random weights of the complex-conjugate eigenvector pair
            _weights = np.random.rand(2) * 2 - 1
            # Make perturbation vector
            perturb = (
                _weights[0] * perturb_e_vectors_conj[:, i // n_runs_per_profile]
                + _weights[1] * perturb_e_vectors[:, i // n_runs_per_profile]
            )

        # Copy array for plotting
        perturb_temp = np.copy(perturb)
        # Find scaling factor in order to have the seeked norm of the error
        lambda_factor = params.seeked_error_norm / np.linalg.norm(perturb)
        # Scale down the perturbation
        perturb = lambda_factor * perturb

        # Perform small test to be noticed if the perturbation is not as expected
        np.testing.assert_almost_equal(
            np.linalg.norm(perturb),
            params.seeked_error_norm,
            decimal=abs(floor(log10(params.seeked_error_norm))) + 1,
        )

        perturbations[params.u_slice, i] = perturb

        if dev_plot_active:
            g_dev_plots.dev_plot_perturbation_generation(perturb, perturb_temp)

    return perturbations
