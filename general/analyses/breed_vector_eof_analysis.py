import sys

sys.path.append("..")
from typing import Tuple
import numpy as np
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.argument_parsers as a_parsers
import general.analyses.plot_analyses as g_plt_analyses
from general.params.model_licences import Models
from config import MODEL


# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params


def calc_bv_eof_vectors(breed_vectors: np.ndarray) -> np.ndarray:
    """Computes an orthogonal complement to the breed vectors

    Parameters
    ----------
    breed_vectors : np.ndarray
        The breed vectors to analyse. Shape: (n_units, n_vectors, sdim)

    Returns
    -------
    np.ndarray
        The resulting EOF vectors. Shape: (n_units, sdim, n_vectors)
    """
    # Get number of vectors
    n_vectors: int = breed_vectors.shape[1]
    n_units: int = breed_vectors.shape[0]
    # Due to the way the breed vectors are stored, the transpose is directly given
    breed_vectors_transpose: np.ndarray = breed_vectors
    breed_vectors: np.ndarray = np.transpose(breed_vectors, axes=(0, 2, 1))

    # Calculate covariance matrix
    cov_matrix: np.ndarray = breed_vectors_transpose @ breed_vectors / n_vectors

    # Calculate eigenvalues and -vectors
    e_values, e_vectors = np.linalg.eig(cov_matrix)

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
    ).real

    return eof_vectors


if __name__ == "__main__":
    # Get arguments
    mult_pert_arg_setup = a_parsers.MultiPerturbationArgSetup()
    mult_pert_arg_setup.setup_parser()
    args = mult_pert_arg_setup.args
    # Get BVs
    perturb_vectors, perturb_header_dicts = pt_import.import_perturb_vectors(args)

    calc_bv_eof_vectors(perturb_vectors)
