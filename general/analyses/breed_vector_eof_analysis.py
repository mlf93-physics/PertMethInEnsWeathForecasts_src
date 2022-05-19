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
import shell_model_experiments.utils.util_funcs as sh_utils
import lorentz63_experiments.params.params as l63_params
import general.utils.importing.import_perturbation_data as pt_import
import shell_model_experiments.utils.special_params as sh_sparams
import lorentz63_experiments.params.special_params as l63_sparams
import general.utils.argument_parsers as a_parsers
import general.utils.saving.save_vector_funcs as v_save
import general.utils.saving.save_data_funcs as g_save
import general.analyses.analyse_data as g_anal
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
        _,
        u_init_profiles,
        eval_pos,
        perturb_header_dicts,
    ) = pt_import.import_perturb_vectors(
        args,
        raw_perturbations=args["bv_raw_perts"],
        force_no_ref_import=args["bv_raw_perts"],
    )

    # Calculate the orthogonal complement to the vectors
    eof_vectors, variances = g_anal.calc_eof_vectors(
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

    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        sh_utils.update_dependent_params(params)
        sh_utils.set_params(params, parameter="sdim", value=args["sdim"])
        sh_utils.update_arrays(params)
        args["ny"] = sh_utils.ny_from_ny_n_and_forcing(
            args["forcing"], args["ny_n"], args["diff_exponent"]
        )

    g_ui.confirm_run_setup(args)

    main(args)
