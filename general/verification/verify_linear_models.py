import sys

sys.path.append("..")
import matplotlib.pyplot as plt
from pyinstrument import Profiler
import numpy as np
import general.utils.importing.import_data_funcs as g_import
import general.utils.saving.save_perturbation as pt_save
import general.utils.running.runner_utils as r_utils
from general.utils.module_import.type_import import *
import general.utils.argument_parsers as a_parsers
from general.params.model_licences import Models
import config as cfg

import matplotlib.pyplot as plt

if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.utils.special_params as sh_sparams
    from shell_model_experiments.params.params import PAR
    from shell_model_experiments.params.params import ParamsStructType
    import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
    import shell_model_experiments.utils.util_funcs as ut_funcs
    from shell_model_experiments.sabra_model.sabra_model import run_model as sh_model
    import shell_model_experiments.utils.runner_utils as sh_r_utils
    from shell_model_experiments.sabra_model.tl_sabra_model import (
        run_model as sh_tl_model,
    )
    from shell_model_experiments.sabra_model.sabra_model_combinations import (
        sh_tl_atl_model,
    )

    # Get parameters for model
    params = PAR
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams
    import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
    import lorentz63_experiments.utils.util_funcs as ut_funcs
    from lorentz63_experiments.lorentz63_model.lorentz63 import run_model as l63_model
    from lorentz63_experiments.lorentz63_model.tl_lorentz63 import (
        run_model as l63_tl_model,
    )
    from lorentz63_experiments.lorentz63_model.lorentz63_model_combinations import (
        l63_tl_atl_model,
    )

    # Get parameters for model
    params = l63_params
    sparams = l63_sparams

profiler = Profiler()


def run_tl_shell_model_verification(
    args,
    u_ref,
    u_perturb,
    tl_data_out,
    nl_model_pert_data_out,
    nl_model_non_pert_data_out,
):
    # Initialise the Jacobian and diagonal arrays
    (
        J_matrix,
        diagonal0,
        diagonal1,
        diagonal2,
        diagonal_1,
        diagonal_2,
    ) = sh_nm_estimator.init_jacobian()

    # Run TL model
    sh_tl_model(
        u_perturb,
        u_ref,
        tl_data_out,
        args["Nt"],
        args["ny"],
        args["diff_exponent"],
        args["forcing"],
        PAR,
        J_matrix,
        diagonal0,
        diagonal1,
        diagonal2,
        diagonal_1,
        diagonal_2,
        raw_perturbation=True,
    )

    # Get diffusion functions
    if args["diff_type"] == "inf_hyper":
        diff_function = ut_funcs.infinit_hyper_diffusion
    else:
        diff_function = ut_funcs.normal_diffusion

    # Run non-linear model on perturbed start velocity
    sh_model(
        diff_function,
        u_ref + u_perturb,
        nl_model_pert_data_out,
        args["Nt"],
        args["ny"],
        args["forcing"],
        args["diff_exponent"],
        PAR,
    )
    # Run non-linear model on non-perturbed start velocity
    sh_model(
        diff_function,
        u_ref,
        nl_model_non_pert_data_out,
        args["Nt"],
        args["ny"],
        args["forcing"],
        args["diff_exponent"],
        PAR,
    )


def run_l63_tl_model_verification(
    args,
    u_ref,
    u_perturb,
    jacobian_matrix,
    lorentz_matrix,
    tl_data_out,
    nl_model_pert_data_out,
    nl_model_non_pert_data_out,
):

    # Run TL model
    l63_tl_model(
        u_perturb,
        u_ref,
        lorentz_matrix,
        jacobian_matrix,
        tl_data_out,
        args["Nt"],
        r_const=args["r_const"],
        raw_perturbation=True,
    )

    # Run non-linear model on perturbed start velocity
    l63_model(
        u_ref + u_perturb,
        lorentz_matrix,
        nl_model_pert_data_out,
        args["Nt"],
    )

    # Run non-linear model on non-perturbed start velocity
    l63_model(
        u_ref,
        lorentz_matrix,
        nl_model_non_pert_data_out,
        args["Nt"],
    )


def run_sh_atl_model_verification(
    args: dict,
    u_ref: np.ndarray,
    u_perturb: np.ndarray,
    data_out: np.ndarray,
    J_matrix: np.ndarray,
    diagonal0: np.ndarray,
    diagonal1: np.ndarray,
    diagonal2: np.ndarray,
    diagonal_1: np.ndarray,
    diagonal_2: np.ndarray,
):
    u_tl_out, u_atl_out, u_init_perturb = sh_tl_atl_model(
        u_perturb,
        u_ref,
        data_out,
        args,
        J_matrix,
        diagonal0,
        diagonal1,
        diagonal2,
        diagonal_1,
        diagonal_2,
    )

    rhs_identity = np.dot(u_init_perturb[sparams.u_slice].conj(), u_atl_out)
    lhs_identity = np.dot(u_tl_out.conj(), u_tl_out)
    diff_identity = np.abs(lhs_identity - rhs_identity) / (
        np.abs(1 / 2 * (lhs_identity + rhs_identity))
    )

    return diff_identity


def run_l63_atl_model_verification(
    args, u_ref, u_perturb, jacobian_matrix, lorentz_matrix, data_out
):
    u_tl_out, u_atl_out, u_init_perturb = l63_tl_atl_model(
        u_perturb, u_ref, jacobian_matrix, lorentz_matrix, data_out, args
    )

    rhs_identity = np.dot(u_init_perturb, u_atl_out)
    lhs_identity = np.dot(u_tl_out, u_tl_out)

    diff_identity = abs(lhs_identity - rhs_identity) / np.mean(
        [lhs_identity, rhs_identity]
    )

    return diff_identity


def verify_tlm_model(args: dict):
    """Verify the TLM of the current model, by calculating the error between
    the TLM solution and the difference between a perturbed full model run and
    an unperturbed full model run.

    Parameters
    ----------
    args : dict
        The run-time arguments

    Raises
    ------
    g_exceptions.ModelError
        Raised if running on the shell model - verification not implemented yet
    """
    # Import reference data
    u_ref, _, ref_header_dict = g_import.import_start_u_profiles(args=args)

    # Prepare random perturbation
    perturbations, perturb_positions, _ = r_utils.prepare_perturbations(
        args, raw_perturbations=True
    )
    # Normalize perturbation
    # u_perturb = g_utils.normalize_array(perturb, norm_value=seeked_error_norm)
    error_data_out = np.empty(
        (
            int(args["Nt"] * params.sample_rate),
            params.sdim + 1,
        ),
        dtype=sparams.dtype,
    )
    tl_data_out = np.empty(
        (int(round(args["Nt"] * params.sample_rate)), params.sdim + 1),
        dtype=sparams.dtype,
    )
    nl_model_pert_data_out = np.empty(
        (int(round(args["Nt"] * params.sample_rate)), params.sdim + 1),
        dtype=sparams.dtype,
    )
    nl_model_non_pert_data_out = np.empty(
        (int(round(args["Nt"] * params.sample_rate)), params.sdim + 1),
        dtype=sparams.dtype,
    )

    if cfg.MODEL == Models.SHELL_MODEL:
        print(f"\nRunning verification of the Sabra TL model\n")
        # Run verification multiple times
        for i in range(int(args["n_profiles"] * args["n_runs_per_profile"])):
            run_tl_shell_model_verification(
                args,
                np.copy(u_ref[:, i]),
                np.copy(perturbations[:, i]),
                tl_data_out,
                nl_model_pert_data_out,
                nl_model_non_pert_data_out,
            )

            error_data_out[:, 1:] = tl_data_out[:, 1:] - (
                nl_model_pert_data_out[:, 1:] - nl_model_non_pert_data_out[:, 1:]
            )
            # Insert time
            error_data_out[:, 0] = tl_data_out[:, 0]

            pt_save.save_perturbation_data(
                error_data_out,
                perturb_position=perturb_positions[i // args["n_runs_per_profile"]],
                perturb_count=i,
                run_count=0,
                args=args,
            )
    elif cfg.MODEL == Models.LORENTZ63:
        # Initialise jacobian and deriv matrix
        jacobian_matrix = l63_nm_estimator.init_jacobian(args)
        lorentz_matrix = ut_funcs.setup_lorentz_matrix(args)

        print(f"\nRunning verification of the Lorentz63 TL model\n")
        # Run verification multiple times
        for i in range(int(args["n_profiles"] * args["n_runs_per_profile"])):

            run_l63_tl_model_verification(
                args,
                np.copy(u_ref[:, i]),
                np.copy(perturbations[:, i]),
                jacobian_matrix,
                lorentz_matrix,
                tl_data_out,
                nl_model_pert_data_out,
                nl_model_non_pert_data_out,
            )

            error_data_out[:, 1:] = tl_data_out[:, 1:] - (
                nl_model_pert_data_out[:, 1:] - nl_model_non_pert_data_out[:, 1:]
            )
            error_data_out[:, 0] = tl_data_out[:, 0]

            pt_save.save_perturbation_data(
                error_data_out,
                perturb_position=perturb_positions[i // args["n_runs_per_profile"]],
                perturb_count=i,
                run_count=0,
                args=args,
            )


def verify_atlm_model(args: dict):
    # Set number of iterations to a low number, e.g. to investigate one
    # iteration
    if cfg.MODEL == cfg.Models.LORENTZ63:
        args["Nt"] = 500
    elif cfg.MODEL == cfg.Models.SHELL_MODEL:
        args["Nt"] = 1
    # Import reference data
    u_ref, _, ref_header_dict = g_import.import_start_u_profiles(args=args)

    # Prepare random perturbation
    # perturb = pt_utils.generate_rd_perturbations()
    perturbations, _, _ = r_utils.prepare_perturbations(args, raw_perturbations=True)
    # Prepare arrays
    data_out = np.zeros(
        (args["Nt"] + args["endpoint"] * 1, params.sdim + 1), dtype=sparams.dtype
    )
    # Run verification multiple times
    n_runs = int(args["n_profiles"] * args["n_runs_per_profile"])
    diff_identity_array = np.empty(n_runs, dtype=np.float64)

    if cfg.MODEL == Models.SHELL_MODEL:
        # Set samplerate
        ut_funcs.set_params(PAR, parameter="sample_rate", value=1.0)
        # Initialise the Jacobian and diagonal arrays
        (
            J_matrix,
            diagonal0,
            diagonal1,
            diagonal2,
            diagonal_1,
            diagonal_2,
        ) = sh_nm_estimator.init_jacobian()

        print(f"\nRunning verification of the ATL Sabra shell model\n")

        for i in range(n_runs):
            diff_identity = run_sh_atl_model_verification(
                args,
                np.copy(u_ref[:, i]),
                np.copy(perturbations[:, i]),
                data_out,
                J_matrix,
                diagonal0,
                diagonal1,
                diagonal2,
                diagonal_1,
                diagonal_2,
            )

            diff_identity_array[i] = diff_identity

    elif cfg.MODEL == Models.LORENTZ63:

        jacobian_matrix = l63_nm_estimator.init_jacobian(args)
        lorentz_matrix = ut_funcs.setup_lorentz_matrix(args)

        print(f"\nRunning verification of the Lorentz63 ATL model\n")

        for i in range(n_runs):

            diff_identity = run_l63_atl_model_verification(
                args,
                np.copy(u_ref[:, i]),
                np.copy(perturbations[:, i]),
                jacobian_matrix,
                lorentz_matrix,
                data_out,
            )

            diff_identity_array[i] = diff_identity

    zero_entries = diff_identity_array == 0

    logged_diff_identity_array = np.log(
        diff_identity_array, where=np.logical_not(zero_entries)
    )
    logged_diff_identity_array[zero_entries] = None
    mean_diff_identity = np.nanmean(logged_diff_identity_array)
    std_diff_identity = np.nanstd(logged_diff_identity_array)

    print(f"{cfg.MODEL} | V_atlm = {mean_diff_identity} +/- {std_diff_identity}")

    # plt.plot(logged_diff_identity_array)
    # plt.plot([0, n_runs], [mean_diff_identity, mean_diff_identity], "k--")
    # plt.xlabel("Profile index")
    # plt.ylabel("Logged error rel. mean of identity")
    # plt.title(f"Verification of ATLM | $N_{{iterations}}$={int(args['Nt'])}")
    # plt.show()


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    mult_pert_arg_setup = a_parsers.MultiPerturbationArgSetup()
    mult_pert_arg_setup.setup_parser()
    ref_arg_setup = a_parsers.ReferenceAnalysisArgParser()
    ref_arg_setup.setup_parser()
    verif_arg_setup = a_parsers.VerificationArgParser()
    verif_arg_setup.setup_parser()
    args = verif_arg_setup.args

    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        ut_funcs.update_dependent_params(PAR)
        ut_funcs.set_params(PAR, parameter="sdim", value=args["sdim"])
        ut_funcs.update_arrays(PAR)
        args["ny"] = ut_funcs.ny_from_ny_n_and_forcing(
            args["forcing"], args["ny_n"], args["diff_exponent"]
        )
        if args["regime_start"] is not None:
            start_times, _, _ = sh_r_utils.get_regime_start_times(args)
            args["start_times"] = start_times[: args["n_profiles"]]

    # Add/edit arguments
    args["Nt"] = int(round(args["time_to_run"] / params.dt))

    profiler.start()

    if args["verification_type"] == "verify_tlm":
        verify_tlm_model(args)
    if args["verification_type"] == "verify_atlm":
        verify_atlm_model(args)

    profiler.stop()
    print(profiler.output_text(color=True))
