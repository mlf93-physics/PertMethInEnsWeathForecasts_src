import sys

sys.path.append("..")
import numpy as np
from shell_model_experiments.params.params import *


# @jit((types.Array(types.complex128, 2, 'C', readonly=True),
#        types.Array(types.complex128, 2, 'C', readonly=False),
#        types.Array(types.complex128, 2, 'C', readonly=False),
#        types.boolean, types.int64, types.float64), parallel=True, cache=True)
def find_normal_modes(
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
