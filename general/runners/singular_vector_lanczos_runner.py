"""Calculate the singular vectors

Example
-------
python ../general/runners/singular_vector_runner.py --exp_setup=TestRun0 --n_units=1

"""
import sys

sys.path.append("..")
from pyinstrument import Profiler

profiler = Profiler()

import copy
import pathlib as pl

import config as cfg
import general.runners.perturbation_runner as pt_runner
import general.runners.iterative_unit_runner as it_unit_runner
import general.utils.argument_parsers as a_parsers
import general.utils.experiments.exp_utils as exp_utils
import general.utils.experiments.validate_exp_setups as ut_exp_val
import general.utils.importing.import_data_funcs as g_import
import general.utils.perturb_utils as pt_utils
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

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.utils.special_params as sh_sparams
    import shell_model_experiments.utils.util_funcs as sh_utils
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType

    from shell_model_experiments.sabra_model.sabra_model import run_model as sh_model

    params = PAR_SH
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams

    params = l63_params
    sparams = l63_sparams

# Set global params
cfg.GLOBAL_PARAMS.ref_run = False


def main(args: dict, exp_setup: dict = None):
    if exp_setup is None:
        # Set exp_setup path
        exp_file_path = pl.Path(
            "./params/experiment_setups/singular_vector_experiment_setups.json"
        )
        # Get the current experiment setup
        exp_setup = exp_utils.get_exp_setup(exp_file_path, args)

    # Get number of existing units
    n_existing_units = lib_file_utils.count_existing_files_or_dirs(
        search_path=pl.Path(args["datapath"], exp_setup["folder_name"]),
        search_pattern="singular_vector*.csv",
    )

    # Validate the start time method
    ut_exp_val.validate_start_time_method(exp_setup=exp_setup)

    # Since the SV calculation is an iterative procedure for a given unit, and
    # because the units are calculated in parallel, n_profiles = n_units in order
    # for the import of start velocity profiles to work.
    if args["n_units"] < np.inf:
        args["n_profiles"] = args["n_units"]
    else:
        args["n_units"] = args["n_profiles"]

    # Generate start times
    start_times, num_possible_units = r_utils.generate_start_times(exp_setup, args)
    # Get index numbers of units to generate
    unit_indices = np.arange(
        n_existing_units,
        min(args["n_units"] + n_existing_units, num_possible_units),
        dtype=np.int32,
    )
    # Filter out all start_times which are not to be used
    start_times = [start_times[index] for index in unit_indices]

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

    # Set error norm to 1
    _temp_seeked_error_norm = copy.deepcopy(params.seeked_error_norm)
    if cfg.MODEL == Models.SHELL_MODEL:
        sh_utils.set_params(params, parameter="seeked_error_norm", value=1)
    elif cfg.MODEL == Models.LORENTZ63:
        params.seeked_error_norm = 1

    # Update start times
    copy_args_tl["start_times"] = start_times
    copy_args_atl["start_times"] = [
        start_times[i] + exp_setup["integration_time"]
        for i in range(args["n_profiles"])
    ]

    # Import start profiles
    u_ref, _, ref_header_dict = g_import.import_start_u_profiles(args=copy_args_tl)

    # Make processes
    processes = []
    processes, data_out_dict = it_unit_runner.main_setup(
        args=copy_args_tl,
        exp_setup=exp_setup,
        u_ref=u_ref,
        n_existing_units=n_existing_units,
    )

    # Run processes
    pr_utils.main_run(processes)

    # Set out folder
    args["out_exp_folder"] = pl.Path(exp_setup["folder_name"])

    # Save singular vectors
    for unit in data_out_dict.keys():
        v_save.save_vector_unit(
            data_out_dict[unit]["sv_matrix"].T,
            characteristic_values=data_out_dict[unit]["s_values"],
            perturb_position=int(round(start_times[unit] * params.tts)),
            unit=data_out_dict[unit]["unit_count"],
            args=args,
            exp_setup=exp_setup,
        )

    # Reset exp_folder
    args["out_exp_folder"] = exp_setup["folder_name"]
    # Save exp setup to exp folder
    g_save.save_exp_info(exp_setup, args)

    # Reset seeked error norm
    if cfg.MODEL == Models.SHELL_MODEL:
        sh_utils.set_params(
            params, parameter="seeked_error_norm", value=_temp_seeked_error_norm
        )
    elif cfg.MODEL == Models.LORENTZ63:
        params.seeked_error_norm = _temp_seeked_error_norm

    if args["erda_run"]:
        path = pl.Path(args["datapath"], exp_setup["folder_name"])
        g_save_utils.compress_dir(path)


if __name__ == "__main__":
    profiler.start()
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

    # Add ny argument
    if cfg.MODEL == Models.SHELL_MODEL:
        args["ny"] = sh_utils.ny_from_ny_n_and_forcing(
            args["forcing"], args["ny_n"], args["diff_exponent"]
        )
    main(args)

    profiler.stop()
    print(profiler.output_text(color=True))
