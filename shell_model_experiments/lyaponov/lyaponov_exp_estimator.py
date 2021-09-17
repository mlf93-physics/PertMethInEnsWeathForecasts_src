import os
import sys

sys.path.append("..")
from math import floor, log10
import numpy as np
from numba import jit, types
from shell_model_experiments.params.params import *
from shell_model_experiments.utils.dev_plots import (
    dev_plot_perturbation_generation,
)

# @jit((types.Array(types.complex128, 2, 'C', readonly=True),
#        types.Array(types.complex128, 2, 'C', readonly=False),
#        types.Array(types.complex128, 2, 'C', readonly=False),
#        types.boolean, types.int64, types.float64), parallel=True, cache=True)
def find_eigenvector_for_perturbation(
    u_init_profiles, dev_plot_active=False, n_profiles=None, local_ny=None
):
    """Find the eigenvector corresponding to the minimal of the positive
    eigenvalues of the initial vel. profile.

    Use the form of the sabra model to perform the calculation of the Jacobian
    matrix. Perform singular-value-decomposition to get the eigenvalues and
    -vectors. Choose the minimal of the positive eigenvalues with respect to
    the real part of the eigenvalue.

    Parameters
    ----------
    u_init_profiles : ndarray
        The initial velocity profiles

    Returns
    -------
    max_e_vector : ndarray
        The eigenvectors corresponding to the minimal of the positive eigenvalues

    """
    print(
        "\nFinding the eigenvalues and eigenvectors at the position of the"
        + " given velocity profiles\n"
    )

    # Prepare for returning all eigen vectors and values
    e_vector_collection = []
    e_value_collection = []

    e_vector_matrix = np.zeros((n_k_vec, n_profiles), dtype=np.complex128)

    # Perform the conjugation
    u_init_profiles_conj = u_init_profiles.conj()
    # Prepare prefactor vector to multiply on J_matrix
    prefactor_reshaped = np.reshape(pre_factor, (-1, 1))
    # Perform calculation for all u_profiles
    for i in range(n_profiles):
        # Calculate the Jacobian matrix
        J_matrix = np.zeros((n_k_vec, n_k_vec), dtype=np.complex128)
        # Add k=2 diagonal
        J_matrix += np.diag(u_init_profiles_conj[bd_size + 1 : -bd_size - 1, i], k=2)
        # Add k=1 diagonal
        J_matrix += factor2 * np.diag(
            np.concatenate(
                (np.array([0 + 0j]), u_init_profiles_conj[bd_size : -bd_size - 2, i])
            ),
            k=1,
        )
        # Add k=-1 diagonal
        J_matrix += factor3 * np.diag(
            np.concatenate(
                (np.array([0 + 0j]), u_init_profiles[bd_size : -bd_size - 2, i])
            ),
            k=-1,
        )
        # Add k=-2 diagonal
        J_matrix += factor3 * np.diag(
            u_init_profiles[bd_size + 1 : -bd_size - 1, i], k=-2
        )

        # Add contribution from derivatives of the complex conjugates:
        J_matrix += np.diag(
            np.concatenate(
                (u_init_profiles[bd_size + 2 : -bd_size, i], np.array([0 + 0j]))
            ),
            k=1,
        )
        J_matrix += factor2 * np.diag(
            np.concatenate(
                (u_init_profiles[bd_size + 2 : -bd_size, i], np.array([0 + 0j]))
            ),
            k=-1,
        )

        J_matrix = J_matrix * prefactor_reshaped

        # Add the k=0 diagonal
        # temp_ny = args['ny'] if header is None else header['ny']
        J_matrix -= np.diag(local_ny * k_vec_temp ** 2, k=0)

        e_values, e_vectors = np.linalg.eig(J_matrix)

        e_vector_collection.append(e_vectors)
        e_value_collection.append(e_values)

        # positive_e_values_indices = np.argwhere(e_values.real > 0)
        chosen_e_value_index = np.argmax(e_values.real)

        e_vector_matrix[:, i] = e_vectors[:, chosen_e_value_index]
        J_matrix.fill(0 + 0j)

        # if dev_plot_active:
        #     print('Largest positive eigenvalue', e_values[chosen_e_value_index])

        #     dev_plot_eigen_mode_analysis(e_values, J_matrix, e_vectors,
        #         header=header, perturb_pos=perturb_positions[i])

    return e_vector_matrix, e_vector_collection, e_value_collection


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
        (n_k_vec + 2 * bd_size, n_profiles * n_runs_per_profile), dtype=np.complex128
    )

    if args["eigen_perturb"]:
        # Get complex-conjugate vector pair
        perturb_e_vectors_conj = np.conj(perturb_e_vectors)

    # Perform perturbation for all eigenvectors
    for i in range(n_profiles * n_runs_per_profile):
        # Apply single shell perturbation
        if args["single_shell_perturb"] is not None:
            perturb = np.zeros(n_k_vec, dtype=np.complex128)
            perturb.real[args["single_shell_perturb"]] = (
                np.random.rand(1)[0].astype(np.float64) * 2 - 1
            )
            perturb.imag[args["single_shell_perturb"]] = (
                np.random.rand(1)[0].astype(np.float64) * 2 - 1
            )
        elif not args["eigen_perturb"]:
            # Generate random perturbation error
            # Reshape into complex array
            perturb = np.empty(n_k_vec, dtype=np.complex128)
            # Generate random error
            error = np.random.rand(2 * n_k_vec).astype(np.float64) * 2 - 1
            perturb.real = error[:n_k_vec]
            perturb.imag = error[n_k_vec:]
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
        lambda_factor = seeked_error_norm / np.linalg.norm(perturb)
        # Scale down the perturbation
        perturb = lambda_factor * perturb

        # Perform small test to be noticed if the perturbation is not as expected
        np.testing.assert_almost_equal(
            np.linalg.norm(perturb),
            seeked_error_norm,
            decimal=abs(floor(log10(seeked_error_norm))) + 1,
        )

        perturbations[bd_size:-bd_size, i] = perturb

        if dev_plot_active:
            dev_plot_perturbation_generation(perturb, perturb_temp)

    return perturbations
