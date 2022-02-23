import config as cfg
import general.utils.perturb_utils as pt_utils
from general.params.model_licences import Models
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
    # Initiate rescaled_perturbations
    lanczos_outarray = None

    # Initiate the Lanczos arrays and algorithm
    propagated_vector: np.ndarray((params.sdim, 1)) = np.zeros(
        (params.sdim, 1), dtype=sparams.dtype
    )
    input_vector: np.ndarray((params.sdim, 1)) = np.zeros(
        (params.sdim, 1), dtype=sparams.dtype
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
        # Start out by running on u_old, but hereafter lanczos_outarray will be
        # updated by the lanczos_iterator
        lanczos_outarray = u_old
        lanczos_iterator = pt_utils.lanczos_vector_algorithm(
            propagated_vector=propagated_vector,
            input_vector_j=input_vector,
            n_iterations=exp_setup["n_vectors"],
        )
        # Calculate the desired number of SVs
        for _ in range(exp_setup["n_vectors"]):
            # Run specified number of model iterations
            for _ in range(exp_setup["n_model_iterations"]):
                if cfg.MODEL == Models.SHELL_MODEL:
                    _, u_atl_out, u_init_perturb = sh_tl_atl_model(
                        lanczos_outarray.ravel().conj(),
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
                    # Run TL model followed by ATL model
                    _, u_atl_out, u_init_perturb = l63_tl_atl_model(
                        lanczos_outarray.ravel(),
                        u_ref,
                        jacobian_matrix,
                        lorentz_matrix,
                        data_out,
                        copy_args,
                    )

                #  Rescale perturbations
                # lanczos_outarray = pt_utils.rescale_perturbations(
                #     u_atl_out[np.newaxis, :], copy_args, raw_perturbations=True
                # )
            # Update arrays for the lanczos algorithm
            propagated_vector[:, :] = np.reshape(u_atl_out, (params.sdim, 1))
            input_vector[:, :] = u_init_perturb[sparams.u_slice, np.newaxis]

            # Iterate the Lanczos algorithm one step
            lanczos_outarray, tridiag_matrix, input_vector_matrix = next(
                lanczos_iterator
            )

            # NOTE: Uncomment this to enable normalization of lanczos vectors -
            # produces orthogonal lanczos vectors. lanczos_outarray =
            # g_utils.normalize_array( lanczos_outarray,
            #     norm_value=params.seeked_error_norm, axis=0 )
            lanczos_outarray = np.pad(
                lanczos_outarray,
                pad_width=((params.bd_size, params.bd_size), (0, 0)),
                mode="constant",
            )

        # Calculate SVs from eigen vectors of tridiag_matrix
        temp_sv_matrix, temp_s_values = pt_utils.calculate_svs(
            tridiag_matrix, input_vector_matrix
        )

        # Add to arrays to perform averaging
        sv_matrix += temp_sv_matrix
        s_values += temp_s_values

    # Average singular vectors and values
    sv_matrix /= exp_setup["n_lanczos_iterations"]
    s_values /= exp_setup["n_lanczos_iterations"]

    return sv_matrix, s_values
