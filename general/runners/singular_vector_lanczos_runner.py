"""Calculate the singular vectors

Example
-------
python ../general/runners/singular_vector_runner.py --exp_setup=TestRun0 --n_units=1

"""

from pyinstrument import Profiler

profiler = Profiler()
profiler.start()
import sys

sys.path.append("..")
import copy
import pathlib as pl

import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.experiments.exp_utils as exp_utils
import general.utils.experiments.validate_exp_setups as ut_exp_val
import general.utils.importing.import_data_funcs as g_import
import general.utils.runner_utils as r_utils
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_utils as g_save_utils
import general.utils.saving.save_vector_funcs as v_save
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
import general.utils.perturb_utils as pt_utils
import lorentz63_experiments.params.params as l63_params
import numpy as np
import shell_model_experiments.utils.util_funcs as sh_utils
from general.params.model_licences import Models
import shell_model_experiments.utils.special_params as sh_sparams
import lorentz63_experiments.params.special_params as l63_sparams
from shell_model_experiments.params.params import PAR as PAR_SH
from shell_model_experiments.params.params import ParamsStructType

import perturbation_runner as pt_runner


# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    params = PAR_SH
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    params = l63_params
    sparams = l63_sparams

# Set global params
cfg.GLOBAL_PARAMS.ref_run = False


def main(args):
    # Set exp_setup path
    exp_file_path = pl.Path(
        "./params/experiment_setups/singular_vector_experiment_setups.json"
    )
    # Get the current experiment setup
    exp_setup = exp_utils.get_exp_setup(exp_file_path, args)

    # Get number of existing blocks
    n_existing_units = g_utils.count_existing_files_or_dirs(
        search_path=pl.Path(args["datapath"], exp_setup["folder_name"]),
        search_pattern="singular_vector*.csv",
    )

    # Validate the start time method
    ut_exp_val.validate_start_time_method(exp_setup=exp_setup)

    # Generate start times
    start_times, num_possible_units = r_utils.generate_start_times(exp_setup, args)

    processes = []
    # Prepare arguments
    args["pert_mode"] = "rd"
    args["time_to_run"] = exp_setup["integration_time"]
    args["start_time_offset"] = (
        exp_setup["vector_offset"] if "vector_offset" in exp_setup else None
    )
    args["endpoint"] = True
    # Set exp folder (folder is reset before save of exp info)
    args["out_exp_folder"] = pl.Path(
        exp_setup["folder_name"], exp_setup["sub_exp_folder"]
    )
    # Adjust start times
    args = g_utils.adjust_start_times_with_offset(args)

    # Skip data save
    args["skip_save_data"] = True
    # Copy args for models
    copy_args_tl = copy.deepcopy(args)
    copy_args_atl = copy.deepcopy(args)

    # Calculate the desired number of units
    for i in range(
        n_existing_units,
        min(args["n_units"] + n_existing_units, num_possible_units),
    ):
        # Update start times
        copy_args_tl["start_times"] = [start_times[i]]
        copy_args_atl["start_times"] = [start_times[i] + exp_setup["integration_time"]]

        # Prepare storage of sv vectors and -values
        sv_matrix_store = np.empty(
            (exp_setup["n_iterations"], params.sdim, exp_setup["n_vectors"])
        )
        s_values_store = np.empty((exp_setup["n_iterations"], exp_setup["n_vectors"]))

        for j in range(exp_setup["n_iterations"]):
            # Import reference data
            u_ref, _, ref_header_dict = g_import.import_start_u_profiles(
                args=copy_args_tl
            )

            # Initiate rescaled_perturbations
            lanczos_outarray = None

            # Initiate the Lanczos arrays and algorithm
            propagated_vector: np.ndarray((params.sdim, 1)) = np.zeros((params.sdim, 1))
            input_vector: np.ndarray((params.sdim, 1)) = np.zeros((params.sdim, 1))
            lanczos_iterator = pt_utils.lanczos_vector_algorithm(
                propagated_vector=propagated_vector,
                input_vector_j=input_vector,
                n_iterations=exp_setup["n_vectors"],
            )

            for _ in range(exp_setup["n_vectors"]):
                ###### TL model run ######
                # Add TL submodel attribute
                cfg.MODEL.submodel = "TL"
                # On all other iterations except the first, the rescaled
                # perturbations are used and are not None
                (
                    processes,
                    data_out_list,
                    _,
                    u_profiles_perturbed,
                ) = pt_runner.main_setup(
                    copy_args_tl,
                    u_profiles_perturbed=lanczos_outarray,
                    exp_setup=exp_setup,
                    u_ref=u_ref,
                )

                if len(processes) > 0:
                    # Run specified number of cycles
                    pt_runner.main_run(
                        processes,
                        args=copy_args_tl,
                    )
                else:
                    print("No processes to run - check if units already exists")

                ###### Adjoint model run ######
                # Add ATL submodel attribute
                cfg.MODEL.submodel = "ATL"
                processes, data_out_list, _, _ = pt_runner.main_setup(
                    copy_args_atl,
                    u_profiles_perturbed=np.array(data_out_list).T,
                    exp_setup=exp_setup,
                    u_ref=u_ref,
                )

                if len(processes) > 0:
                    # Run specified number of iterations
                    pt_runner.main_run(
                        processes,
                        args=copy_args_atl,
                    )

                else:
                    print("No processes to run - check if units already exists")

                # Update arrays for the lanczos algorithm
                propagated_vector[:, :] = u_profiles_perturbed
                input_vector[:, :] = np.reshape(data_out_list[0], (params.sdim, 1))
                # Iterate the Lanczos algorithm one step
                lanczos_outarray, tridiag_matrix, input_vector_matrix = next(
                    lanczos_iterator
                )

            # Calculate SVs from eigen vectors of tridiag_matrix
            sv_matrix, s_values = pt_utils.calculate_svs(
                tridiag_matrix, input_vector_matrix
            )

            sv_matrix_store[j, :, :] = sv_matrix
            s_values_store[j, :] = s_values

        # Set out folder
        args["out_exp_folder"] = pl.Path(exp_setup["folder_name"])
        # Reset submodel
        cfg.MODEL.submodel = None

        # Do averaging
        sv_matrix_average = np.mean(sv_matrix_store, axis=0)
        s_values_average = np.mean(s_values_store, axis=0)

        # Save singular vectors
        v_save.save_vector_unit(
            sv_matrix_average.T,
            characteristic_values=s_values_average,
            perturb_position=int(round(start_times[i] * params.tts)),
            unit=i,
            args=args,
            exp_setup=exp_setup,
        )

    # Reset exp_folder
    args["out_exp_folder"] = exp_setup["folder_name"]
    # Save exp setup to exp folder
    g_save.save_exp_info(exp_setup, args)

    if args["erda_run"]:
        path = pl.Path(args["datapath"], exp_setup["folder_name"])
        g_save_utils.compress_dir(path)


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    mult_pert_arg_setup = a_parsers.MultiPerturbationArgSetup()
    mult_pert_arg_setup.setup_parser()
    ref_arg_setup = a_parsers.ReferenceAnalysisArgParser()
    ref_arg_setup.setup_parser()
    args = ref_arg_setup.args

    g_ui.confirm_run_setup(args)
    r_utils.adjust_run_setup(args)

    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        sh_utils.update_dependent_params(params, sdim=int(args["sdim"]))
        sh_utils.update_arrays(params)

    # Add ny argument
    if cfg.MODEL == Models.SHELL_MODEL:
        args["ny"] = sh_utils.ny_from_ny_n_and_forcing(
            args["forcing"], args["ny_n"], args["diff_exponent"]
        )
    main(args)

    profiler.stop()
    # print(profiler.output_text(color=True))
