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

    # Prepare prefactor vector to multiply on J_matrix
    pre_factor_reshaped = np.reshape(PAR.pre_factor, (-1, 1))
    # Perform calculation for all u_profiles
    for i in range(args["n_profiles"]):
        # Calculate the Jacobian matrix
        # J_matrix = np.zeros((PAR.sdim, PAR.sdim), dtype=sparams.dtype)
        J_matrix = calc_jacobian(
            np.copy(u_init_profiles[:, i]),
            args["diff_exponent"],
            local_ny,
            pre_factor_reshaped,
            PAR.sdim,
            PAR.k_vec_temp,
        )

        e_values, e_vectors = np.linalg.eig(J_matrix)

        e_vector_collection.append(e_vectors)
        e_value_collection.append(e_values)

        # positive_e_values_indices = np.argwhere(e_values.real > 0)
        chosen_e_value_index = np.argmax(e_values.real)

        e_vector_matrix[:, i] = e_vectors[:, chosen_e_value_index]

        # if dev_plot_active:
        #     print('Largest positive eigenvalue', e_values[chosen_e_value_index])

        #     dev_plot_eigen_mode_analysis(e_values, J_matrix, e_vectors,
        #         header=header, perturb_pos=perturb_positions[i])

    return e_vector_matrix, e_vector_collection, e_value_collection


@nb.njit(
    (nb.types.Array(nb.types.complex128, 2, "C", readonly=False))(
        nb.types.Array(nb.types.complex128, 1, "C", readonly=True),
        nb.types.float64,
        nb.types.float64,
        nb.types.Array(nb.types.complex128, 2, "C", readonly=True),
        nb.typeof(PAR),
    ),
    cache=cfg.NUMBA_CACHE,
)
def calc_jacobian(
    ref_u_vector: np.ndarray,
    diff_exponent,
    local_ny,
    pre_factor_reshaped,
    PAR,
):
    # Perform the conjugation
    ref_u_vector_conj: np.ndarray = ref_u_vector.conj()
    # Initialise the Jacobian
    J_matrix = np.zeros((PAR.sdim, PAR.sdim), dtype=sparams.dtype)

    # Add k=2 diagonal
    J_matrix += np.diag(ref_u_vector_conj[PAR.bd_size + 1 : -PAR.bd_size - 1], k=2)
    # Add k=1 diagonal
    J_matrix += PAR.factor2 * np.diag(
        np.concatenate(
            (np.array([0 + 0j]), ref_u_vector_conj[PAR.bd_size : -PAR.bd_size - 2])
        ),
        k=1,
    )
    # Add k=-1 diagonal
    J_matrix += PAR.factor3 * np.diag(
        np.concatenate(
            (np.array([0 + 0j]), ref_u_vector[PAR.bd_size : -PAR.bd_size - 2])
        ),
        k=-1,
    )
    # Add k=-2 diagonal
    J_matrix += PAR.factor3 * np.diag(
        ref_u_vector[PAR.bd_size + 1 : -PAR.bd_size - 1], k=-2
    )

    # Add contribution from derivatives of the complex conjugates:
    J_matrix += np.diag(
        np.concatenate(
            (ref_u_vector[PAR.bd_size + 2 : -PAR.bd_size], np.array([0 + 0j]))
        ),
        k=1,
    )
    J_matrix += PAR.factor2 * np.diag(
        np.concatenate(
            (ref_u_vector[PAR.bd_size + 2 : -PAR.bd_size], np.array([0 + 0j]))
        ),
        k=-1,
    )

    J_matrix = J_matrix * pre_factor_reshaped

    # Add the k=0 diagonal
    J_matrix -= np.diag(local_ny * PAR.k_vec_temp ** diff_exponent, k=0)

    return J_matrix
