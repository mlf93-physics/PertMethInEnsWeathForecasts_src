import sys

sys.path.append("..")
from math import floor, log10
import numpy as np
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
from general.utils.module_import.type_import import *
import general.utils.dev_plots as g_dev_plots
import general.utils.importing.import_data_funcs as g_import
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.util_funcs as g_utils
from general.params.model_licences import Models
from config import MODEL

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params


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
    perturbations : ndarray
        The random perturbations
    """
    n_profiles = args["n_profiles"]
    n_runs_per_profile = args["n_runs_per_profile"]
    perturbations = np.zeros(
        (params.sdim + 2 * params.bd_size, n_profiles * n_runs_per_profile),
        dtype=params.dtype,
    )

    if args["pert_mode"] == "nm":
        # Get complex-conjugate vector pair
        perturb_vectors_conj = np.conj(perturb_vectors)

    # Perform perturbation for all eigenvectors
    for i in range(n_profiles * n_runs_per_profile):
        if args["pert_mode"] is not None:
            # Apply random perturbation
            if args["pert_mode"] == "rd":
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

            # Apply normal mode perturbation
            elif args["pert_mode"] == "nm":
                # Generate random weights of the complex-conjugate eigenvector pair
                _weights = np.random.rand(2) * 2 - 1
                # Make perturbation vector
                perturb = (
                    _weights[0] * perturb_vectors_conj[:, i // n_runs_per_profile]
                    + _weights[1] * perturb_vectors[:, i // n_runs_per_profile]
                ).real

            # Apply breed vector perturbation
            elif args["pert_mode"] == "bv":
                # Generate random weights of the complex-conjugate eigenvector pair
                # _weight = np.random.rand() * 2 - 1
                # Make perturbation vector
                perturb = perturb_vectors[:, i]

            # Apply breed vector EOF perturbations
            elif args["pert_mode"] == "bv_eof":
                perturb = perturb_vectors[:, i]

        # Apply single shell perturbation
        elif args["single_shell_perturb"] is not None:
            perturb = np.zeros(params.sdim, dtype=params.dtype)
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

        perturbations[params.u_slice, i] = perturb

        if dev_plot_active:
            g_dev_plots.dev_plot_perturbation_generation(perturb, perturb_temp)

    return perturbations


def rescale_perturbations(perturb_data, args):

    num_perturbations = args["n_runs_per_profile"]

    # Import reference data
    (
        u_init_profiles,
        _,
        _,
    ) = g_import.import_start_u_profiles(args=args)

    # Transform into 2d array
    perturb_data = np.array(perturb_data)
    # Pad array if necessary
    perturb_data = np.pad(
        perturb_data,
        pad_width=((0, 0), (params.bd_size, params.bd_size)),
        mode="constant",
    )

    # Diff data
    diff_data = perturb_data.T - u_init_profiles
    # Rescale data
    rescaled_data = (
        diff_data
        / np.reshape(np.linalg.norm(diff_data, axis=0), (1, num_perturbations))
        * params.seeked_error_norm
    )

    # Add rescaled data to u_init_profiles
    u_init_profiles += rescaled_data

    return u_init_profiles


# NOTE Remove the following code - unused.
# def prepare_breed_vectors(args: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Import and prepare the breed vectors

#     Parameters
#     ----------
#     args : dict
#         Run-time arguments

#     Returns
#     -------
#     Tuple[np.ndarray, np.ndarray, np.ndarray]
#         (
#             u_init_profiles : The vel. profiles at the validation time
#             eval_positions : The index position of the evaluation time
#             breed_vectors : The breed vectors
#         )

#     Raises
#     ------
#     ValueError
#         Raised if no breed vectors was imported
#     """
#     # Get BVs
#     breed_vectors, breed_vector_header_dicts = pt_import.import_perturb_vectors(args)

#     if breed_vectors.size == 0:
#         raise ValueError("No perturbation vectors imported")

#     # Reshape and transpose
#     breed_vectors = np.reshape(
#         breed_vectors,
#         (args["n_profiles"] * args["n_runs_per_profile"], params.sdim),
#     ).T

#     # Prepare start times for import
#     args["start_times"] = [
#         breed_vector_header_dicts[i]["val_pos"] * params.stt
#         for i in range(len(breed_vector_header_dicts))
#     ]

#     # Import start u profiles
#     (
#         u_init_profiles,
#         eval_positions,
#         header_dict,
#     ) = g_import.import_start_u_profiles(args=args)

#     return u_init_profiles, eval_positions, breed_vectors
