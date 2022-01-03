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
    from shell_model_experiments.sabra_model.tl_sabra_model import (
        run_model as sh_tl_model,
    )
    from shell_model_experiments.sabra_model.atl_sabra_model import (
        run_model as sh_atl_model,
    )

    params = PAR_SH
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams
    import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
    import lorentz63_experiments.utils.util_funcs as ut_funcs
    from lorentz63_experiments.lorentz63_model.tl_lorentz63 import (
        run_model as l63_tl_model,
    )
    from lorentz63_experiments.lorentz63_model.atl_lorentz63 import (
        run_model as l63_atl_model,
    )

    params = l63_params
    sparams = l63_sparams


def sv_generator(
    exp_setup,
    copy_args,
    u_ref,
    u_old: np.ndarray,
    run_count: int,
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
    lanczos_iterator = pt_utils.lanczos_vector_algorithm(
        propagated_vector=propagated_vector,
        input_vector_j=input_vector,
        n_iterations=exp_setup["n_vectors"],
    )

    sv_matrix = np.zeros(
        (params.sdim, exp_setup["n_vectors"]),
        dtype=sparams.dtype,
    )
    s_values = np.zeros(
        exp_setup["n_vectors"],
        dtype=np.complex128,
    )
    # Average over multiple iterations of the lanczos algorithm
    for _ in range(exp_setup["n_lanczos_iterations"]):
        # Calculate the desired number of SVs
        for _ in range(exp_setup["n_vectors"]):
            # Run specified number of model iterations
            for k in range(exp_setup["n_model_iterations"]):
                if cfg.MODEL == Models.SHELL_MODEL:
                    print(
                        "Optimized SV calculation for shell model not implemented yet"
                    )
                elif cfg.MODEL == Models.LORENTZ63:

                    # Setup matrixes
                    lorentz_matrix = ut_funcs.setup_lorentz_matrix(copy_args)
                    jacobian_matrix = l63_nm_estimator.init_jacobian(copy_args)

                    l63_tl_model(
                        u_old,
                        np.copy(u_ref[:, run_count]),
                        lorentz_matrix,
                        jacobian_matrix,
                        data_out,
                        copy_args["Nt"] + copy_args["endpoint"] * 1,
                        r_const=copy_args["r_const"],
                        raw_perturbation=True,
                    )

                    # Store the initial perturbation vector
                    if k == 0:
                        store_u_profiles_perturbed = np.copy(u_old[sparams.u_slice])

                    l63_atl_model(
                        np.reshape(data_out[-1, 1:].T, (params.sdim, 1)),
                        np.copy(u_ref[:, run_count]),
                        lorentz_matrix,
                        jacobian_matrix,
                        data_out,
                        copy_args["Nt"] + copy_args["endpoint"] * 1,
                        r_const=copy_args["r_const"],
                        raw_perturbation=True,
                    )

                # NOTE: Rescale perturbations
                lanczos_outarray = pt_utils.rescale_perturbations(
                    data_out[0, 1:], copy_args, raw_perturbations=True
                )

            # Update arrays for the lanczos algorithm
            propagated_vector[:, :] = np.reshape(data_out[0, 1:], (params.sdim, 1))
            input_vector[:, :] = store_u_profiles_perturbed
            # Iterate the Lanczos algorithm one step
            lanczos_outarray, tridiag_matrix, input_vector_matrix = next(
                lanczos_iterator
            )
            lanczos_outarray = np.pad(
                lanczos_outarray,
                pad_width=((params.bd_size, params.bd_size), (0, 0)),
                mode="constant",
            )

        # Calculate SVs from eigen vectors of tridiag_matrix
        temp_sv_matrix, temp_s_values = pt_utils.calculate_svs(
            tridiag_matrix, input_vector_matrix
        )

        sv_matrix += temp_sv_matrix
        s_values += temp_s_values

    # Average singular vectors and values
    sv_matrix /= exp_setup["n_lanczos_iterations"]
    s_values /= exp_setup["n_lanczos_iterations"]

    return sv_matrix, s_values
