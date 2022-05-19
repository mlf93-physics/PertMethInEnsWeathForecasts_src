import config as cfg
import general.utils.perturb_utils as pt_utils
from general.params.model_licences import Models
from general.params.experiment_licences import Experiments as EXP
import general.analyses.plot_analyses as g_plt_anal
import numpy as np

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.utils.special_params as sh_sparams
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType
    import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
    import shell_model_experiments.utils.util_funcs as ut_funcs
    from shell_model_experiments.sabra_model.sabra_model_combinations import (
        sh_tl_atl_model,
        sh_atl_tl_model,
    )

    params = PAR_SH
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams
    import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
    import lorentz63_experiments.utils.util_funcs as ut_funcs
    from lorentz63_experiments.lorentz63_model.lorentz63_model_combinations import (
        l63_tl_atl_model,
        l63_atl_tl_model,
    )

    params = l63_params
    sparams = l63_sparams


def sv_generator(
    exp_setup,
    copy_args,
    u_ref,
    u_old: np.ndarray,
    data_out: np.ndarray,
):

    # Initiate the Lanczos arrays and algorithm
    propagated_vector: np.ndarray(params.sdim) = np.zeros(
        params.sdim, dtype=sparams.dtype
    )
    lanczos_vector_matrix: np.ndarray = np.zeros(
        (params.sdim, exp_setup["n_vectors"]), dtype=sparams.dtype
    )

    sv_matrix = np.zeros(
        (params.sdim, exp_setup["n_vectors"]),
        dtype=sparams.dtype,
    )
    s_values = np.zeros(
        exp_setup["n_vectors"],
        dtype=np.complex128,
    )

    # Setup matrixes
    if cfg.MODEL == Models.SHELL_MODEL:
        # Initialise the Jacobian and diagonal arrays
        (
            J_matrix,
            diagonal0,
            diagonal1,
            diagonal2,
            diagonal_1,
            diagonal_2,
        ) = sh_nm_estimator.init_jacobian()
    elif cfg.MODEL == Models.LORENTZ63:
        lorentz_matrix = ut_funcs.setup_lorentz_matrix(copy_args)
        jacobian_matrix = l63_nm_estimator.init_jacobian(copy_args)

    # Average over multiple iterations of the lanczos algorithm
    for _ in range(exp_setup["n_lanczos_iterations"]):
        # Start out by running on u_old, but hereafter lanczos_vector_matrix will be
        # updated by the lanczos_iterator
        lanczos_vector_matrix[:, 0] = u_old[sparams.u_slice]
        lanczos_iterator = pt_utils.lanczos_vector_algorithm(
            propagated_vector=propagated_vector,
            lanczos_vector_matrix=lanczos_vector_matrix,
            n_iterations=exp_setup["n_vectors"],
        )
        # Calculate the desired number of SVs
        for i in range(exp_setup["n_vectors"]):
            # Run specified number of model iterations
            for _ in range(exp_setup["n_model_iterations"]):
                if cfg.MODEL == Models.SHELL_MODEL:
                    if cfg.LICENCE == EXP.SINGULAR_VECTORS:
                        _, u_atl_out, _ = sh_tl_atl_model(
                            np.pad(
                                lanczos_vector_matrix[:, i],
                                pad_width=params.bd_size,  # , params.bd_size), (0, 0)),
                                mode="constant",
                            ),
                            u_ref,
                            data_out,
                            copy_args,
                            J_matrix,
                            diagonal0,
                            diagonal1,
                            diagonal2,
                            diagonal_1,
                            diagonal_2,
                        )
                    elif cfg.LICENCE == EXP.FINAL_SINGULAR_VECTORS:
                        u_tl_out, _, _ = sh_atl_tl_model(
                            np.pad(
                                lanczos_vector_matrix[:, i],
                                pad_width=params.bd_size,  # , params.bd_size), (0, 0)),
                                mode="constant",
                            ),
                            u_ref,
                            data_out,
                            copy_args,
                            J_matrix,
                            diagonal0,
                            diagonal1,
                            diagonal2,
                            diagonal_1,
                            diagonal_2,
                        )
                elif cfg.MODEL == Models.LORENTZ63:
                    if cfg.LICENCE == EXP.SINGULAR_VECTORS:
                        # Run TL model followed by ATL model
                        _, u_atl_out, _ = l63_tl_atl_model(
                            np.copy(lanczos_vector_matrix[:, i]),
                            u_ref,
                            jacobian_matrix,
                            lorentz_matrix,
                            data_out,
                            copy_args,
                        )
                    elif cfg.LICENCE == EXP.FINAL_SINGULAR_VECTORS:
                        # Run TL model followed by ATL model
                        u_tl_out, _, _ = l63_atl_tl_model(
                            np.copy(lanczos_vector_matrix[:, i]),
                            u_ref,
                            jacobian_matrix,
                            lorentz_matrix,
                            data_out,
                            copy_args,
                        )

                #  Rescale perturbations
                # lanczos_vector_next = pt_utils.rescale_perturbations(
                #     u_atl_out[np.newaxis, :], copy_args, raw_perturbations=True
                # )
            # Update array for the lanczos algorithm
            if cfg.LICENCE == EXP.SINGULAR_VECTORS:
                propagated_vector[:] = u_atl_out
            elif cfg.LICENCE == EXP.FINAL_SINGULAR_VECTORS:
                propagated_vector[:] = u_tl_out

            # Iterate the Lanczos algorithm one step
            tridiag_matrix = next(lanczos_iterator)

        # Calculate orthogonality between Lanczos vectors
        orthogonality = g_plt_anal.orthogonality_of_vectors(lanczos_vector_matrix)
        # print("orthogonality", orthogonality)
        # print("tridiag_matrix", tridiag_matrix)
        # print("lanczos_vector_matrix", lanczos_vector_matrix)

        # Calculate SVs from eigen vectors of tridiag_matrix
        temp_sv_matrix, temp_s_values = pt_utils.calculate_svs(
            tridiag_matrix, lanczos_vector_matrix
        )

        # Add to arrays to perform averaging
        sv_matrix += temp_sv_matrix
        s_values += temp_s_values

    # Average singular vectors and values
    sv_matrix /= exp_setup["n_lanczos_iterations"]
    s_values /= exp_setup["n_lanczos_iterations"]

    return sv_matrix, s_values
