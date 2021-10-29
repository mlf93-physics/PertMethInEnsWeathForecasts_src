import sys

sys.path.append("..")
import numpy as np
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.argument_parsers as a_parsers
import general.utils.user_interface as g_ui
from general.params.model_licences import Models
import config as cfg


# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    params = sh_params
elif cfg.MODEL == Models.LORENTZ63:
    params = l63_params


def calc_bv_eof_vectors(breed_vectors: np.ndarray, n_eof_vectors: int) -> np.ndarray:
    """Computes an orthogonal complement to the breed vectors

    Parameters
    ----------
    breed_vectors : np.ndarray
        The breed vectors to analyse. Shape: (n_units, n_vectors, sdim)

    Returns
    -------
    np.ndarray
        The resulting EOF vectors. Shape: (n_units, sdim, n_eof_vectors)
    """
    # Get number of vectors
    n_vectors: int = breed_vectors.shape[1]
    n_units: int = breed_vectors.shape[0]

    # Test dimensions against n_eof_vectors requested
    if n_vectors < n_eof_vectors:
        raise ValueError(
            "The number of requested EOF vectors exceeds the number of breed vectors; "
            + f"number of BVs: {n_vectors}, number of requested EOFs: {n_eof_vectors}"
        )
    # Due to the way the breed vectors are stored, the transpose is directly given
    breed_vectors_transpose: np.ndarray = breed_vectors
    breed_vectors: np.ndarray = np.transpose(breed_vectors, axes=(0, 2, 1))

    # Calculate covariance matrix
    cov_matrix: np.ndarray = breed_vectors_transpose @ breed_vectors / n_vectors

    # Calculate eigenvalues and -vectors
    e_values, e_vectors = np.linalg.eig(cov_matrix)
    # Take absolute in order to avoid negative variances
    e_values = np.abs(e_values)

    sort_indices = np.argsort(e_values, axis=1)[:, ::-1]

    # Project e vectors onto the breed vectors to get the EOF vectors
    eof_vectors: np.ndarray((n_units, params.sdim, n_vectors)) = (
        breed_vectors
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

    # Get sorted vectors
    eof_vectors = eof_vectors[np.arange(n_units)[:, np.newaxis], :, sort_indices]
    eof_vectors = np.transpose(eof_vectors, axes=(0, 2, 1))

    # Filter out unused eof's
    eof_vectors = eof_vectors[:, :, :n_eof_vectors]

    return eof_vectors


if __name__ == "__main__":
    cfg.init_licence()
    # Get arguments
    mult_pert_arg_setup = a_parsers.MultiPerturbationArgSetup()
    mult_pert_arg_setup.setup_parser()
    args = mult_pert_arg_setup.args

    g_ui.confirm_run_setup(args)

    # Get BVs
    (
        vector_units,
        u_init_profiles,
        eval_pos,
        perturb_header_dicts,
    ) = pt_import.import_perturb_vectors(args)

    eof_vectors = calc_bv_eof_vectors(vector_units, 2)
