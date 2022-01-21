import sys

sys.path.append("..")
import numpy as np
import numba as nb
from general.utils.module_import.type_import import *
from shell_model_experiments.params.params import PAR, ParamsStructType
import shell_model_experiments.utils.special_params as sparams
import config as cfg


def find_normal_modes(
    u_init_profiles: np.ndarray,
    args: dict,
    dev_plot_active: bool = False,
    local_ny: float = None,
) -> Tuple[np.ndarray, list, list]:
    """Find the eigenvector corresponding to the minimal of the positive
    eigenvalues of the initial vel. profile.

    Use the form of the sabra model to perform the calculation of the Jacobian
    matrix. Perform singular-value-decomposition to get the eigenvalues and
    -vectors. Choose the minimal of the positive eigenvalues with respect to
    the real part of the eigenvalue.

    Parameters
    ----------
    u_init_profiles : np.ndarray
        The initial velocity profiles
    args : dict
        Run-time arguments
    dev_plot_active : bool, optional
        Make development plots or not, by default False
    local_ny : float, optional
        Local value of ny, by default None

    Returns
    -------
    Tuple[np.ndarray, list, list]
        (
            e_vector_matrix : The eigen vectors corresponding to the maximum
                eigen value collected in a matrix
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

    e_vector_matrix = np.zeros((PAR.sdim, args["n_profiles"]), dtype=sparams.dtype)

    # Initialise the Jacobian and diagonal arrays
    (
        J_matrix,
        diagonal0,
        diagonal1,
        diagonal2,
        diagonal_1,
        diagonal_2,
    ) = init_jacobian()

    # Perform calculation for all u_profiles
    for i in range(args["n_profiles"]):
        # Calculate the Jacobian matrix
        calc_jacobian(
            np.copy(u_init_profiles[:, i]),
            args["diff_exponent"],
            local_ny,
            PAR,
            diagonal0,
            diagonal1,
            diagonal2,
            diagonal_1,
            diagonal_2,
        )

        e_values, e_vectors = np.linalg.eig(J_matrix)

        sort_index = np.argsort(e_values.real)[::-1]
        e_vectors = e_vectors[:, sort_index]
        e_values = e_values[sort_index]
        e_vector_matrix[:, i] = e_vectors[:, 0]

        e_vector_collection.append(e_vectors)
        e_value_collection.append(e_values)

    return e_vector_matrix, e_vector_collection, e_value_collection


def init_jacobian():

    # Initialise the Jacobian
    J_matrix = np.zeros((PAR.sdim, PAR.sdim), dtype=sparams.dtype)

    diagonal0 = np.diagonal(J_matrix, 0)
    diagonal1 = np.diagonal(J_matrix, 1)
    diagonal2 = np.diagonal(J_matrix, 2)
    diagonal_1 = np.diagonal(J_matrix, -1)
    diagonal_2 = np.diagonal(J_matrix, -2)

    diagonal0.setflags(write=True)
    diagonal1.setflags(write=True)
    diagonal2.setflags(write=True)
    diagonal_1.setflags(write=True)
    diagonal_2.setflags(write=True)

    return J_matrix, diagonal0, diagonal1, diagonal2, diagonal_1, diagonal_2


@nb.njit(
    (
        nb.types.Array(nb.types.complex128, 1, "C", readonly=True),
        nb.types.float64,
        nb.types.float64,
        nb.typeof(PAR),
        nb.types.Array(nb.types.complex128, 1, "A", readonly=False),
        nb.types.Array(nb.types.complex128, 1, "A", readonly=False),
        nb.types.Array(nb.types.complex128, 1, "A", readonly=False),
        nb.types.Array(nb.types.complex128, 1, "A", readonly=False),
        nb.types.Array(nb.types.complex128, 1, "A", readonly=False),
    ),
    cache=cfg.NUMBA_CACHE,
)
def calc_jacobian(
    ref_u_vector: np.ndarray,
    diff_exponent: float,
    local_ny: float,
    PAR: ParamsStructType,
    diagonal0: np.ndarray,
    diagonal1: np.ndarray,
    diagonal2: np.ndarray,
    diagonal_1: np.ndarray,
    diagonal_2: np.ndarray,
):
    # Start by resetting the diagonals
    diagonal0.fill(0 + 0j)
    diagonal1.fill(0 + 0j)
    diagonal2.fill(0 + 0j)
    diagonal_1.fill(0 + 0j)
    diagonal_2.fill(0 + 0j)

    # Perform the conjugation
    ref_u_vector_conj: np.ndarray = ref_u_vector.conj()

    # Add k=2 diagonal
    diagonal2 += ref_u_vector_conj[PAR.bd_size + 1 : -PAR.bd_size - 1]
    # Add k=1 diagonal
    diagonal1[1:] += PAR.factor2 * ref_u_vector_conj[PAR.bd_size : -PAR.bd_size - 2]

    # Add k=-1 diagonal
    diagonal_1[1:] += PAR.factor3 * ref_u_vector[PAR.bd_size : -PAR.bd_size - 2]
    # Add k=-2 diagonal
    diagonal_2 += PAR.factor3 * ref_u_vector[PAR.bd_size + 1 : -PAR.bd_size - 1]

    # Add contribution from derivatives of the complex conjugates:
    diagonal1[:-1] += ref_u_vector[PAR.bd_size + 2 : -PAR.bd_size]
    diagonal_1[:-1] += PAR.factor2 * ref_u_vector[PAR.bd_size + 2 : -PAR.bd_size]

    # Multiply common prefactor
    diagonal0 *= PAR.pre_factor
    diagonal1 *= PAR.pre_factor[:-1]
    diagonal2 *= PAR.pre_factor[:-2]
    diagonal_1 *= PAR.pre_factor[1:]
    diagonal_2 *= PAR.pre_factor[2:]

    # Add the k=0 diagonal
    diagonal0 -= local_ny * PAR.k_vec_temp ** diff_exponent


@nb.njit(
    (
        nb.types.Array(nb.types.complex128, 1, "C", readonly=True),
        nb.types.float64,
        nb.types.float64,
        nb.typeof(PAR),
        nb.types.Array(nb.types.complex128, 2, "C", readonly=False),
        nb.types.Array(nb.types.complex128, 1, "A", readonly=False),
        nb.types.Array(nb.types.complex128, 1, "A", readonly=False),
        nb.types.Array(nb.types.complex128, 1, "A", readonly=False),
        nb.types.Array(nb.types.complex128, 1, "A", readonly=False),
        nb.types.Array(nb.types.complex128, 1, "A", readonly=False),
    ),
    cache=cfg.NUMBA_CACHE,
)
def calc_adjoint_jacobian(
    ref_u_vector: np.ndarray,
    diff_exponent: float,
    local_ny: float,
    PAR: ParamsStructType,
    J_matrix: np.ndarray,
    diagonal0: np.ndarray,
    diagonal1: np.ndarray,
    diagonal2: np.ndarray,
    diagonal_1: np.ndarray,
    diagonal_2: np.ndarray,
):

    calc_jacobian(
        ref_u_vector,
        diff_exponent,
        local_ny,
        PAR,
        diagonal0,
        diagonal1,
        diagonal2,
        diagonal_1,
        diagonal_2,
    )

    return J_matrix.T.conj()
