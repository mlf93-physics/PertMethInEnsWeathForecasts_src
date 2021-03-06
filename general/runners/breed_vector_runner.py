"""
Example
-------
python ../general/runners/breed_vector_runner.py --exp_setup=TestRun2 --n_units=1 --pert_mode=rd --start_times=11.2
"""
import sys

sys.path.append("..")
import copy
import numpy as np
import pathlib as pl

import config as cfg
import general.runners.perturbation_runner as pt_runner
import general.utils.argument_parsers as a_parsers
import general.utils.experiments.exp_utils as exp_utils
import general.utils.experiments.validate_exp_setups as ut_exp_val
import general.utils.perturb_utils as pt_utils
import general.utils.running.runner_utils as r_utils
import general.utils.process_utils as pr_utils
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_utils as g_save_utils
import general.utils.saving.save_vector_funcs as v_save
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
from libs.libutils import file_utils as lib_file_utils
from general.params.model_licences import Models
from pyinstrument import Profiler

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    from shell_model_experiments.params.params import ParamsStructType
    from shell_model_experiments.params.params import PAR as PAR_SH
    import shell_model_experiments.utils.util_funcs as sh_utils
    import shell_model_experiments.utils.special_params as sh_sparams
    import shell_model_experiments.utils.runner_utils as sh_r_utils

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
    """Run the BREED VECTOR experiment to generate breed vectors based on the
    reference records

    Parameters
    ----------
    args : dict
        Run-time arguments
    exp_setup : dict, optional
        The experiment setup, by default None
    """
    if exp_setup is None:
        # Set exp_setup path
        exp_file_path = pl.Path(
            "./params/experiment_setups/breed_vector_experiment_setups.json"
        )
        # Get the current experiment setup
        exp_setup = exp_utils.get_exp_setup(exp_file_path, args)

    # Get number of existing blocks
    n_existing_units = lib_file_utils.count_existing_files_or_dirs(
        search_path=pl.Path(args["datapath"], exp_setup["folder_name"]),
        search_pattern="breeding_vector*.csv",
    )

    # Validate the start time method
    ut_exp_val.validate_start_time_method(exp_setup=exp_setup)

    # Only generate start times if not requesting regime start
    if args["regime_start"] is None:
        # Generate start times
        start_times, num_possible_units = r_utils.generate_start_times(exp_setup, args)
    elif cfg.MODEL == Models.SHELL_MODEL:
        start_times, num_possible_units, _ = sh_r_utils.get_regime_start_times(args)
        start_times = r_utils.get_bv_lv_start_time(
            eval_time=np.array(start_times), exp_setup=exp_setup
        )

    processes = []

    # Calculate the desired number of units
    for i in range(
        n_existing_units,
        min(args["n_units"] + n_existing_units, num_possible_units),
    ):

        # Make analysis forecasts
        args["time_to_run"] = exp_setup["integration_time"]
        args["start_times"] = [start_times[i]]
        args["start_time_offset"] = (
            exp_setup["vector_offset"] if "vector_offset" in exp_setup else None
        )
        args["endpoint"] = True
        args["n_profiles"] = 1
        args["n_runs_per_profile"] = exp_setup["n_vectors"]
        args["out_exp_folder"] = pl.Path(
            exp_setup["folder_name"], exp_setup["sub_exp_folder"]
        )
        args = g_utils.adjust_start_times_with_offset(args)

        # Copy args in order not override in forecast processes
        copy_args = copy.deepcopy(args)

        if copy_args["save_last_pert"]:
            copy_args["skip_save_data"] = True

        # Start off with None value in order to invoke random perturbations
        rescaled_data = None
        perturb_positions = None
        for j in range(exp_setup["n_cycles"]):

            if copy_args["save_last_pert"] and (j + 1) == exp_setup["n_cycles"]:
                copy_args["skip_save_data"] = False

            processes, data_out_list, perturb_positions, _ = pt_runner.main_setup(
                copy_args,
                u_profiles_perturbed=rescaled_data,
                perturb_positions=perturb_positions,
                exp_setup=exp_setup,
            )

            if len(processes) > 0:
                # Run specified number of cycles
                pr_utils.main_run(
                    processes,
                )
                # Offset time to prepare for next run and import of reference data
                # for rescaling
                copy_args["start_times"][0] += copy_args["time_to_run"]

                # The rescaled data is used to start off cycle 1+
                rescaled_data = pt_utils.rescale_perturbations(
                    data_out_list,
                    copy_args,
                )
                # Update perturb_positions
                perturb_positions += int(exp_setup["integration_time"] * params.tts)
        else:
            print("No processes to run - check if units already exists")

        # Set out folder
        args["out_exp_folder"] = pl.Path(exp_setup["folder_name"])

        rescaled_data = pt_utils.rescale_perturbations(
            data_out_list, copy_args, return_raw_perturbations=args["bv_raw_perts"]
        )
        # Save breed vector data
        v_save.save_vector_unit(
            rescaled_data[sparams.u_slice, :].T,
            perturb_position=int(round(start_times[i] * params.tts)),
            unit=i,
            args=args,
            exp_setup=exp_setup,
        )

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
    args = mult_pert_arg_setup.args

    # Shell model specific
    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        sh_utils.update_dependent_params(params)
        sh_utils.set_params(params, parameter="sdim", value=args["sdim"])
        sh_utils.update_arrays(params)
        # Add ny argument
        args["ny"] = sh_utils.ny_from_ny_n_and_forcing(
            args["forcing"], args["ny_n"], args["diff_exponent"]
        )

    g_ui.confirm_run_setup(args)
    r_utils.adjust_run_setup(args)

    # Make profiler
    profiler = Profiler()
    # Start profiler
    profiler.start()
    main(args)
    profiler.stop()
    print(profiler.output_text(color=True))
