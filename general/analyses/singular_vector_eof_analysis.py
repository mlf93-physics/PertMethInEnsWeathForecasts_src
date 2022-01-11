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
import general.analyses.analyse_data as g_anal
import general.utils.user_interface as g_ui
from general.params.model_licences import Models
import config as cfg


# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.utils.util_funcs as sh_utils

    params = PAR_SH
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    params = l63_params
    sparams = l63_sparams


def main(args: dict, exp_setup: dict = None):
    """Run analysis of SV-EOF vectors on the basis of SV vectors

    Parameters
    ----------
    args : dict
        Run-time arguments
    """
    print("\nRunning SV-EOF analysis\n")
    # Get BVs
    (
        vector_units,
        _,
        u_init_profiles,
        eval_pos,
        perturb_header_dicts,
    ) = pt_import.import_perturb_vectors(
        args, raw_perturbations=True, dtype=sparams.dtype
    )

    # Normalize
    vector_units = g_utils.normalize_array(vector_units, norm_value=1, axis=2)

    # Calculate the orthogonal complement to the BVs
    # eof_vectors, variances = g_anal.calc_eof_vectors(
    #     vector_units, n_eof_vectors=vector_units.shape[1]
    # )
    q_matrices = np.empty(vector_units.shape, dtype=sparams.dtype)
    for i in range(args["n_profiles"]):
        q_matrix, r = np.linalg.qr(vector_units[i, :, :])
        q_matrices[i, :, :] = q_matrix

    # import general.analyses.plot_analyses as g_plt_anal
    # import matplotlib.pyplot as plt
    # import seaborn as sb

    # axes1 = plt.axes()

    # sb.heatmap(
    #     np.mean(np.abs(q_matrices), axis=0).T,
    #     cmap="Reds",
    #     ax=axes1,
    #     cbar_kws={
    #         "pad": 0.1,
    #     },
    # )
    # axes1.invert_yaxis()
    # axes1.invert_xaxis()
    # plt.xticks(rotation=0)
    # axes1.yaxis.tick_right()
    # axes1.yaxis.set_label_position("right")
    # axes1.set_xlabel("SV-EOF index")
    # axes1.set_ylabel("Shell index")

    # orthogonality_matrix = 0
    # for i in range(args["n_profiles"]):
    #     orthogonality_matrix += g_plt_anal.orthogonality_of_vectors(q_matrices[i, :, :])

    # orthogonality_matrix /= args["n_profiles"]

    # axes2 = plt.axes()
    # sb.heatmap(
    #     orthogonality_matrix,
    #     ax=axes2,
    #     cmap="Reds",
    #     vmin=0,
    #     vmax=1,
    #     annot=True,
    #     fmt=".1f",
    #     annot_kws={"fontsize": 8},
    #     cbar_kws=dict(use_gridspec=True, label="Orthogonality"),
    # )
    # plt.plot(variances.T)

    # plt.colorbar()
    # plt.show()

    # Save breed vector EOF vectors
    for unit in range(args["n_profiles"]):
        #     out_unit = np.concatenate(
        #         [variances[unit, :][:, np.newaxis], eof_vectors[unit, :, :].T], axis=1
        # )
        v_save.save_vector_unit(
            q_matrices[i, :, :],
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

    # Add ny argument
    if cfg.MODEL == Models.SHELL_MODEL:
        args["ny"] = sh_utils.ny_from_ny_n_and_forcing(
            args["forcing"], args["ny_n"], args["diff_exponent"]
        )

    g_ui.confirm_run_setup(args)

    main(args)
