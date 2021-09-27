import numpy as np
from lorentz63_experiments.params.params import *


def find_normal_modes(u_init_profiles, args, dev_plot_active=False, n_profiles=None):
    """Find the normal modes corresponding to the maximum positive
    eigenvalues of the initial vel. profile.

    Use the form of the lorentz63 model to perform the calculation of the Jacobian
    matrix. Perform singular-value-decomposition to get the eigenvalues and
    -vectors. Choose the maximum positive eigenvalue with respect to
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

    e_vector_matrix = np.zeros((sdim, n_profiles), dtype=np.complex128)
    e_values_max = np.zeros(n_profiles, dtype=np.complex128)

    # Perform calculation for all u_profiles
    for i in range(n_profiles):
        # Calculate the Jacobian matrix
        J_matrix = np.zeros((sdim, sdim), dtype=dtype)
        J_matrix[0, 0] = -args["sigma"]
        J_matrix[0, 1] = args["sigma"]
        J_matrix[1, 0] = args["r_const"] - u_init_profiles[2, i]
        J_matrix[1, 1] = 1
        J_matrix[1, 2] = -u_init_profiles[0, i]
        J_matrix[2, 0] = u_init_profiles[1, i]
        J_matrix[2, 1] = u_init_profiles[0, i]
        J_matrix[2, 2] = -args["b_const"]

        e_values, e_vectors = np.linalg.eig(J_matrix)

        e_vector_collection.append(e_vectors)
        e_value_collection.append(e_values)

        # positive_e_values_indices = np.argwhere(e_values.real > 0)
        chosen_e_value_index = np.argmax(e_values.real)

        e_vector_matrix[:, i] = e_vectors[:, chosen_e_value_index]
        e_values_max[i] = e_values[chosen_e_value_index]
        # Reset matrix
        J_matrix.fill(0)

        # if dev_plot_active:
        #     print('Largest positive eigenvalue', e_values[chosen_e_value_index])

        #     dev_plot_eigen_mode_analysis(e_values, J_matrix, e_vectors,
        #         header=header, perturb_pos=perturb_positions[i])

    return e_vector_matrix, e_values_max, e_vector_collection, e_value_collection
