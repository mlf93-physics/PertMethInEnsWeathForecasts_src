"""
Calculate the BV-EOF vectors on the basis of BV vectors

Example
-------

python ../general/analyses/breed_vector_eof_analysis.py --out_exp_folder=compare_pert_ttr1.0_run2/bv_eof_vectors --n_runs_per_profile=1 --pert_vector_folder=compare_pert_ttr1.0_run2 --exp_folder=breed_vectors --n_profiles=50
"""

import sys

sys.path.append("..")
import numpy as np
from shell_model_experiments.params.params import ParamsStructType
from shell_model_experiments.params.params import PAR as PAR_SH
import lorentz63_experiments.params.params as l63_params
import general.utils.importing.import_perturbation_data as pt_import
import shell_model_experiments.utils.special_params as sh_sparams
import lorentz63_experiments.params.special_params as l63_sparams
import general.utils.argument_parsers as a_parsers
import general.utils.saving.save_vector_funcs as v_save
import general.utils.saving.save_data_funcs as g_save
import general.utils.util_funcs as g_utils
import general.utils.user_interface as g_ui
from general.params.model_licences import Models
import config as cfg


# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    params = PAR_SH
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    params = l63_params
    sparams = l63_sparams


def calc_bv_eof_vectors(
    breed_vectors: np.ndarray, n_eof_vectors: int = 2
) -> np.ndarray:
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
    print("\nCalculating BV-EOF vectors from BV vectors")
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
    # Take absolute in order to avoid negative variances - observed from values
    # very close to 0 but negative - like -1e-25
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

    # Get sorted eof vectors and e values
    eof_vectors = eof_vectors[np.arange(n_units)[:, np.newaxis], :, sort_indices]
    eof_vectors = np.transpose(eof_vectors, axes=(0, 2, 1))
    e_values = e_values[np.arange(n_units)[:, np.newaxis], sort_indices]

    # Normalize e_values
    e_values = g_utils.normalize_array(e_values, norm_value=1, axis=1)

    # Filter out unused eof's
    eof_vectors = eof_vectors[:, :, :n_eof_vectors]

    return eof_vectors, e_values


def main(args: dict, exp_setup: dict = None):
    """Run analysis of BV-EOF vectors on the basis of BV vectors

    Parameters
    ----------
    args : dict
        Run-time arguments
    """
    print("\nRunning BV-EOF analysis\n")
    # Get BVs
    (
        vector_units,
        u_init_profiles,
        eval_pos,
        perturb_header_dicts,
    ) = pt_import.import_perturb_vectors(args)

    # Calculate the orthogonal complement to the BVs
    eof_vectors, variances = calc_bv_eof_vectors(
        vector_units, n_eof_vectors=vector_units.shape[1]
    )

    # Add underlying velocity profiles to the vectors
    # u_init_profiles = np.reshape(
    #     u_init_profiles[sparams.u_slice, :],
    #     (args["n_profiles"], params.sdim, args["n_runs_per_profile"]),
    # )

    # eof_vectors += u_init_profiles

    # Save breed vector EOF vectors
    for unit in range(args["n_profiles"]):
        out_unit = np.concatenate(
            [variances[unit, :][:, np.newaxis], eof_vectors[unit, :, :].T], axis=1
        )
        v_save.save_vector_unit(
            out_unit,
            perturb_position=int(round(perturb_header_dicts[unit]["val_pos"])),
            unit=unit,
            args=args,
        )

    if exp_setup is not None:
        # Save exp setup to exp folder
        g_save.save_exp_info(exp_setup, args)


if __name__ == "__main__":
    cfg.init_licence()
    # Get arguments
    mult_pert_arg_setup = a_parsers.MultiPerturbationArgSetup()
    mult_pert_arg_setup.setup_parser()
    args = mult_pert_arg_setup.args

    g_ui.confirm_run_setup(args)

    main(args)
