import sys

sys.path.append("..")
from math import floor, log10

import config as cfg
import general.utils.dev_plots as g_dev_plots
import general.utils.importing.import_data_funcs as g_import
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.util_funcs as g_utils
import lorentz63_experiments.params.params as l63_params
import lorentz63_experiments.params.special_params as l63_sparams
import numpy as np
import shell_model_experiments.utils.special_params as sh_sparams
from general.params.model_licences import Models
from general.utils.module_import.type_import import *
from shell_model_experiments.params.params import PAR as PAR_SH
from shell_model_experiments.params.params import ParamsStructType

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    params = PAR_SH
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    params = l63_params
    sparams = l63_sparams


def calculate_perturbations(
    perturb_vectors: np.ndarray, dev_plot_active: bool = False, args: dict = None
) -> np.ndarray:
    """Calculate a random perturbation with a specific norm for each profile.

    The norm of the error is defined in the parameter seeked_error_norm

    Parameters
    ----------
    perturb_vectors : ndarray((sdim, args["n_profiles"]*args["n_runs_per_profile"]))
        The vectors along which to perform the perturbations
    dev_plot_active: bool
        If plotting dev plots or not
    args: dict
        Run-time arguments

    Returns
    -------
    perturbations : ndarray((sdim + 2 * params.bd_size, n_profiles * n_runs_per_profile))
        The random perturbations
    """
    n_profiles = args["n_profiles"]
    n_runs_per_profile = args["n_runs_per_profile"]
    perturbations = np.zeros(
        (params.sdim + 2 * params.bd_size, n_profiles * n_runs_per_profile),
        dtype=sparams.dtype,
    )

    if args["pert_mode"] == "nm":
        # Get complex-conjugate vector pair
        perturb_vectors_conj = np.conj(perturb_vectors)

    # Perform perturbation for all eigenvectors
    for i in range(n_profiles * n_runs_per_profile):
        if args["pert_mode"] is not None:
            # Apply random perturbation
            if args["pert_mode"] == "rd":
                perturb = generate_rd_perturbations()

            # Apply normal mode perturbation
            elif args["pert_mode"] == "nm":
                perturb = generate_nm_perturbations(
                    perturb_vectors, perturb_vectors_conj, n_runs_per_profile, i
                )

            # Apply bv, bv_eof or sv perturbation
            elif "bv" in args["pert_mode"] or "sv" in args["pert_mode"]:
                # Make perturbation vector
                perturb = perturb_vectors[:, i]

        # Apply single shell perturbation
        elif args["single_shell_perturb"] is not None:
            perturb = np.zeros(params.sdim, dtype=sparams.dtype)
            perturb.real[args["single_shell_perturb"]] = (
                np.random.rand(1)[0].astype(np.float64) * 2 - 1
            )
            perturb.imag[args["single_shell_perturb"]] = (
                np.random.rand(1)[0].astype(np.float64) * 2 - 1
            )

        # Copy array for plotting
        perturb_temp = np.copy(perturb)
        # Normalize perturbation
        perturb = g_utils.normalize_array(perturb, norm_value=params.seeked_error_norm)

        # Perform small test to be noticed if the perturbation is not as expected
        np.testing.assert_almost_equal(
            np.linalg.norm(perturb),
            params.seeked_error_norm,
            decimal=abs(floor(log10(params.seeked_error_norm))) + 1,
        )

        perturbations[sparams.u_slice, i] = perturb

        if dev_plot_active:
            g_dev_plots.dev_plot_perturbation_generation(perturb, perturb_temp)

    return perturbations


def rescale_perturbations(
    perturb_data: np.ndarray, args: dict, raw_perturbations: bool = False
) -> np.ndarray:
    """Rescale a set of perturbations to the seeked error norm relative to
    the reference data

    Parameters
    ----------
    perturb_data : np.ndarray
        The perturbations that are rescaled
    args : dict
        Run-time arguments
    raw_perturbations : bool, optional
        If the raw perturbations should be returned instead of the perturbations
        added to the u_init_profiles, by default False


    Returns
    -------
    np.ndarray
        The rescaled perturbations added to the reference data
        (if raw_perturbations is True)
    """

    num_perturbations = args["n_runs_per_profile"]

    # Transform into 2d array
    perturb_data = np.array(perturb_data)
    # Pad array if necessary
    perturb_data = np.pad(
        perturb_data,
        pad_width=((0, 0), (params.bd_size, params.bd_size)),
        mode="constant",
    )

    if not raw_perturbations:
        # Import reference data
        (
            u_init_profiles,
            _,
            _,
        ) = g_import.import_start_u_profiles(args=args)

        # Diff data
        diff_data = perturb_data.T - u_init_profiles
    else:
        diff_data = perturb_data.T

    # Rescale data
    rescaled_data = (
        diff_data
        / np.reshape(np.linalg.norm(diff_data, axis=0), (1, num_perturbations))
        * params.seeked_error_norm
    )

    if rescaled_data.size > params.sdim + 2 * params.bd_size:
        print(
            "Norm between first 2 perturbations after rescaling",
            np.linalg.norm(rescaled_data[:, 0] - rescaled_data[:, 1], axis=0),
        )

    if not raw_perturbations:
        # Add rescaled data to u_init_profiles
        u_init_profiles += rescaled_data
    else:
        # Return raw rescaled perturbations
        u_init_profiles = rescaled_data

    return u_init_profiles


def generate_rd_perturbations() -> np.ndarray:
    """Generate RD perturbations according to the model in use

    Returns
    -------
    np.ndarray
        The generated perturbations
    """
    # Generate random perturbation error
    # Reshape into complex array
    perturb = np.empty(params.sdim, dtype=sparams.dtype)
    # Generate random error
    error = np.random.rand(2 * params.sdim).astype(np.float64) * 2 - 1

    if cfg.MODEL == Models.SHELL_MODEL:
        perturb.real = error[: params.sdim]
        perturb.imag = error[params.sdim :]
    elif cfg.MODEL == Models.LORENTZ63:
        perturb = error[: params.sdim]

    return perturb


def generate_nm_perturbations(
    perturb_vectors: np.ndarray,
    perturb_vectors_conj: np.ndarray,
    n_runs_per_profile: int,
    index: int,
) -> np.ndarray:
    """Generate the NM perturbations according to the model in use

    Parameters
    ----------
    perturb_vectors : np.ndarray
        The vectors to be used to generate perturbations from (i.e. eigen vectors in this case)
    perturb_vectors_conj : np.ndarray
        The conjugated perturb vectors
    n_runs_per_profile : int
        The number of runs per profile
    index : int
        The index of the current perturbation vector

    Returns
    -------
    np.ndarray
        The generated perturbations
    """
    # NOTE - for explanation see https://math.stackexchange.com/questions/3847121/the-importance-of-complex-eigenvectors-in-phase-plane-plotting
    if cfg.MODEL == Models.SHELL_MODEL:
        # Generate random weights
        _rand_numbers = np.random.rand(4) * 2 - 1
        _weight = np.empty(2, dtype=sparams.dtype)
        _weight.real = _rand_numbers[:2]
        _weight.imag = _rand_numbers[2:]

        # Make perturbation vector from the complex-conjugate pair
        perturb = (
            _weight[0] * perturb_vectors_conj[:, index // n_runs_per_profile]
            + _weight[1] * perturb_vectors[:, index // n_runs_per_profile]
        )
    elif cfg.MODEL == Models.LORENTZ63:
        # Generate random weight
        _rand_numbers = np.random.rand(2) * 2 - 1
        _weight = complex(_rand_numbers[0], _rand_numbers[1])

        # Make real perturbation vector from the complex-conjugate pair
        perturb = (
            _weight * perturb_vectors_conj[:, index // n_runs_per_profile]
            + _weight.conjugate() * perturb_vectors[:, index // n_runs_per_profile]
        ).real

    return perturb


def lanczos_vector_algorithm(
    propagated_vector: np.ndarray((params.sdim, 1)) = None,
    input_vector_j: np.ndarray((params.sdim, 1)) = None,
    n_iterations: int = 0,
):
    """Execute the Lanczos algorithm to find eigenvectors and -values of the L*L
    matrix, i.e. the singular vectors and values.

    Calculates the tri-diagonal matrix and Lanczos vectors, which can be
    diagonalised to get eigenvalues and eigenvectors. These vectors can be
    projected onto the Lanczos vectors to get the singular vectors of L*L. See
    pt_utils.calculate_svs function for details of the projection.

    Parameters
    ----------
    propagated_vector : np.ndarray, optional
        The propagated vector w defined as w = L*L v, by default None
    input_vector_j : np.ndarray, optional
        The input vector, v, which is propagated into w, by default None
    n_iterations : int, optional
        The number of iterations of the Lanczos algorithm, by default 0.
        Corresponds to the number of singular vectors
        calculated.

    Yields
    ------
    tuple
        (
            output_vector : np.ndarray
                The vector that will be propagated by L*L in next iterations
            tridiag_matrix : np.ndarray
                The tri-diagonal matrix which can be used to solve the eigenvalue
                problem of L*L.
            input_vector_matrix : np.ndarray
                The matrix with columns equal to all input vectors (Lanczos vectors)
                from all iterations
        )
    """
    beta_j = 0
    tridiag_matrix = np.zeros((n_iterations, n_iterations))
    input_vector_matrix = np.zeros((params.sdim, n_iterations))
    iteration = 0

    def iterator(
        beta_j: float = 0,
        iteration: int = 0,
    ):
        """The iterator of the Lanczos algorithm

        Parameters
        ----------
        beta_j : float, optional
            The previous value for the Beta variable, by default 0
        iteration : int, optional
            The iteration number, by default 0

        Returns
        -------
        tuple
            (
                output_vector : np.ndarray
                    The vector to be propagated by L*L
                beta_j : float
                    The new value for the Beta variable
            )
        """
        omega_j_temp = np.matrix(propagated_vector)
        alpha_j = (omega_j_temp.getH() @ input_vector_j)[0, 0]

        # Save alpha_j to tridiag_matrix
        tridiag_matrix[iteration, iteration] = alpha_j

        # Calculate omega_j (on first iteration beta_j == 0)
        omega_j = (
            omega_j_temp
            - alpha_j * input_vector_j
            - beta_j
            * np.reshape(input_vector_matrix[:, iteration - 1], (params.sdim, 1))
        )

        # Update beta_j
        beta_j = np.linalg.norm(omega_j)

        if iteration > 0:
            # Save beta_j to matrix
            tridiag_matrix[iteration - 1, iteration] = tridiag_matrix[
                iteration, iteration - 1
            ] = beta_j

        if beta_j != 0:
            output_vector = omega_j / beta_j
        else:
            print("hello1, implementation lacking")

        return output_vector, beta_j

    while True:
        output_vector, beta_j = iterator(beta_j, iteration)
        input_vector_matrix[:, iteration] = input_vector_j.ravel()

        # Update iteration
        iteration += 1

        yield output_vector, tridiag_matrix, input_vector_matrix


def calculate_svs(
    tridiag_matrix: np.ndarray, vector_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the singular vectors from the tri-diagonal matrix returned by
    the lanczos_vector_algorithm function.

    Parameters
    ----------
    tridiag_matrix : np.ndarray
        The tri-diagonal matrix to be diagonalised
    vector_matrix : np.ndarray
        A matrix with the Lanczos vectors at each column

    Returns
    -------
    tuple
        (
            sv_matrix : np.ndarray
                The sorted singular vectors of the L*L operator
            s_values : np.ndarray
                The sorted singular values of the L*L operator
        )
    """

    # Get eigenvectors from tridiag_matrix.
    e_values, e_vectors = np.linalg.eig(tridiag_matrix)
    # Take sqrt to get singular values of L
    s_values = np.sqrt(e_values.astype(np.complex128))
    # Sort e_values and e_vectors
    sort_index = np.argsort(s_values)[::-1]
    s_values = s_values[sort_index]
    e_vectors = e_vectors[:, sort_index]
    # Get SVs
    sv_matrix = vector_matrix @ e_vectors
    # Normalize
    sv_matrix = g_utils.normalize_array(
        sv_matrix, norm_value=params.seeked_error_norm, axis=1
    )

    return sv_matrix, s_values
