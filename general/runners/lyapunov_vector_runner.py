"""Calculate the Lyapunov vectors

Example
-------
python ../general/runners/lyapunov_vector_runner.py --exp_setup=TestRun4 --n_units=1

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
import general.utils.process_utils as pr_utils
import general.utils.running.runner_utils as r_utils
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_utils as g_save_utils
import general.utils.saving.save_vector_funcs as v_save
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
from libs.libutils import file_utils as lib_file_utils
import numpy as np
from general.params.model_licences import Models

import perturbation_runner as pt_runner

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.utils.util_funcs as sh_utils
    import shell_model_experiments.utils.runner_utils as sh_r_utils
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType

    params = PAR_SH
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params

    params = l63_params

# Set global params
cfg.GLOBAL_PARAMS.ref_run = False


def main(args):
    # Set exp_setup path
    exp_file_path = pl.Path(
        "./params/experiment_setups/lyapunov_vector_experiment_setups.json"
    )
    # Get the current experiment setup
    exp_setup = exp_utils.get_exp_setup(exp_file_path, args)

    # Get number of existing blocks
    n_existing_units = lib_file_utils.count_existing_files_or_dirs(
        search_path=pl.Path(args["datapath"], exp_setup["folder_name"]),
        search_pattern="lyapunov_vector*.csv",
    )

    # Validate the start time method
    ut_exp_val.validate_start_time_method(exp_setup=exp_setup)

    # Only generate start times if not requesting regime start
    if args["regime_start"] is None:
        # Generate start times
        start_times, num_possible_units = r_utils.generate_start_times(exp_setup, args)
    elif cfg.MODEL == Models.SHELL_MODEL:
        start_times, num_possible_units, _ = sh_r_utils.get_regime_start_times(args)

    processes = []
    # Prepare arguments
    args["pert_mode"] = "rd"
    args["time_to_run"] = exp_setup["integration_time"]
    args["start_time_offset"] = (
        exp_setup["vector_offset"] if "vector_offset" in exp_setup else None
    )
    args["endpoint"] = True
    args["n_profiles"] = 1
    args["n_runs_per_profile"] = exp_setup["n_vectors"]

    # Calculate the desired number of units
    for i in range(
        n_existing_units,
        min(args["n_units"] + n_existing_units, num_possible_units),
    ):
        # Update start times
        args["start_times"] = [start_times[i]]
        # Set exp folder (folder is reset before save of exp info)
        args["out_exp_folder"] = pl.Path(
            exp_setup["folder_name"], exp_setup["sub_exp_folder"]
        )
        args = g_utils.adjust_start_times_with_offset(args)

        # Import reference data
        u_ref, _, ref_header_dict = g_import.import_start_u_profiles(args=args)

        # Copy args in order not override in forecast processes
        copy_args = copy.deepcopy(args)

        processes, data_out_list, _, _ = pt_runner.main_setup(
            copy_args, exp_setup=exp_setup, u_ref=u_ref
        )

        if len(processes) > 0:
            # Run specified number of cycles
            pr_utils.main_run(
                processes,
                args=copy_args,
            )

        else:
            print("No processes to run - check if units already exists")

        # Prepare Lyapunov vector data to be saved
        data_out = np.array(data_out_list)
        # Set out folder
        args["out_exp_folder"] = pl.Path(exp_setup["folder_name"])
        # Save lyapunov vectors
        v_save.save_vector_unit(
            data_out,
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
        sh_utils.update_dependent_params(params)
        sh_utils.set_params(params, parameter="sdim", value=args["sdim"])
        sh_utils.update_arrays(params)

    # Add submodel attribute
    cfg.MODEL.submodel = "TL"

    # Add ny argument
    if cfg.MODEL == Models.SHELL_MODEL:
        args["ny"] = sh_utils.ny_from_ny_n_and_forcing(
            args["forcing"], args["ny_n"], args["diff_exponent"]
        )

    main(args)

    profiler.stop()
    print(profiler.output_text(color=True))
