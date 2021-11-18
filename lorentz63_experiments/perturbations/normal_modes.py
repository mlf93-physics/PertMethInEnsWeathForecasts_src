import numpy as np
from numba import njit, types
from general.utils.module_import.type_import import *
from lorentz63_experiments.params.params import *
import config as cfg


def find_normal_modes(
    u_init_profiles: np.ndarray,
    args: dict,
    n_profiles: int = 1,
) -> Tuple[np.ndarray, np.ndarray, list, list]:
    """Find the normal modes corresponding to the maximum positive
    eigenvalues of the initial vel. profile.

    Use the form of the lorentz63 model to perform the calculation of the Jacobian
    matrix. Perform singular-value-decomposition to get the eigenvalues and
    -vectors. Choose the maximum positive eigenvalue with respect to
    the real part of the eigenvalue.

    Parameters
    ----------
    u_init_profiles : np.ndarray
        The initial velocity profiles
    args : dict
        Run-time arguments
    n_profiles : int, optional
        The number of profiles to analyse for normal modes

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, list, list]
        (
            e_vector_matrix : The eigen vectors corresponding to the maximum
                eigen value collected in a matrix
            e_values_max : The maximum eigenvalues
            e_vector_collection : List of all eigen vectors
            e_value_collection : List of all eigen values
        )
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

    # Initialise the jacobian
    j_matrix = init_jacobian(args)

    # Perform calculation for all u_profiles
    for i in range(n_profiles):
        # Calculate the Jacobian matrix
        # np.copy is needed for Numba to work
        calc_jacobian(j_matrix, np.copy(u_init_profiles[:, i]), r_const=args["r_const"])

        e_values, e_vectors = np.linalg.eig(j_matrix)

        e_vector_collection.append(e_vectors)
        e_value_collection.append(e_values)

        # positive_e_values_indices = np.argwhere(e_values.real > 0)
        chosen_e_value_index = np.argmax(e_values.real)

        e_vector_matrix[:, i] = e_vectors[:, chosen_e_value_index]
        e_values_max[i] = e_values[chosen_e_value_index]

    return e_vector_matrix, e_values_max, e_vector_collection, e_value_collection


def init_jacobian(args):

    j_matrix = np.matrix(np.zeros((sdim, sdim), dtype=np.float64))

    j_matrix[0, 0] = -args["sigma"]
    j_matrix[0, 1] = args["sigma"]
    j_matrix[1, 1] = 1
    j_matrix[2, 2] = -args["b_const"]

    return j_matrix


@njit(
    (
        types.Array(types.float64, 2, "C", readonly=False),
        types.Array(types.float64, 1, "C", readonly=True),
        types.float64,
    ),
    cache=cfg.NUMBA_CACHE,
)
def calc_jacobian(j_matrix, u_profile, r_const):
    """Calculate the jacobian at a given point in time given through the u_profile

    Parameters
    ----------
    j_matrix : numpy.ndarray
        The initialized jacobian matrix
    u_profile : numpy.ndarray
        The velocity profile
    args : dict
        Run-time arguments

    Returns
    -------
    numpy.ndarray
        The jacobian
    """
    j_matrix[1, 0] = r_const - u_profile[2]
    j_matrix[1, 2] = -u_profile[0]
    j_matrix[2, 0] = u_profile[1]
    j_matrix[2, 1] = u_profile[0]


# @njit(
#     (
#         types.Array(types.float64, 2, "C", readonly=False),
#         types.Array(types.float64, 1, "C", readonly=True),
#         types.float64,
#     ),
#     cache=cfg.NUMBA_CACHE,
# )
def calc_adjoint_jacobian(j_matrix, u_profile, r_const):
    """Calculate the adjoint jacobian at a given point in time given through
    the u_profile

    Parameters
    ----------
    j_matrix : numpy.ndarray
        The initialized jacobian matrix
    u_profile : numpy.ndarray
        The velocity profile
    args : dict
        Run-time arguments

    Returns
    -------
    numpy.ndarray
        The jacobian
    """
    j_matrix[1, 0] = r_const - u_profile[2]
    j_matrix[1, 2] = -u_profile[0]
    j_matrix[2, 0] = u_profile[1]
    j_matrix[2, 1] = u_profile[0]

    return j_matrix.getH()
