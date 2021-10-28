import sys

sys.path.append("..")
import pathlib as pl
import decimal
import general.utils.importing.import_data_funcs as g_import
from general.params.experiment_licences import Experiments as EXP
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_utils as g_save_utils
import general.runners.perturbation_runner as pt_runner
import general.utils.exceptions as g_exceptions
import config as cfg


def generate_start_times(exp_setup: dict, args: dict):
    """Generate start times and calculate the number of possible units from
    the relevant run-time variables and variables from the experiment setup

    Parameters
    ----------
    exp_setup : dict
        The current experiment setup
    args : dict
        Run-time arguments

    Returns
    -------
    tuple
        All generated start times and the number of possible units/start times
        (
            list: start_times
            int: num_possible_units
        )

    Raises
    ------
    g_exceptions.LicenceImplementationError
        Raised if the present function do not work on the current licence
    """
    ref_header_dict = g_import.import_info_file(pl.Path(args["datapath"], "ref_data"))

    if cfg.LICENCE == EXP.LORENTZ_BLOCK:
        offset_var = "block_offset"
    elif cfg.LICENCE == EXP.BREEDING_VECTORS or cfg.LICENCE == EXP.LYAPUNOV_VECTORS:
        offset_var = "vector_offset"
    else:
        raise g_exceptions.LicenceImplementationError(licence=cfg.LICENCE)

    if offset_var in exp_setup:
        if "start_times" in exp_setup:
            _time_offset = exp_setup["start_times"][0]
        elif "eval_times" in exp_setup:
            if cfg.LICENCE == EXP.BREEDING_VECTORS:
                _time_offset = (
                    exp_setup["eval_times"][0]
                    - exp_setup["n_cycles"] * exp_setup["integration_time"]
                )
            elif cfg.LICENCE == EXP.LYAPUNOV_VECTORS:
                _time_offset = (
                    exp_setup["eval_times"][0] - exp_setup["integration_time"]
                )
        else:
            _time_offset = 0

        # Determine precision of time
        _precision = (
            decimal.Decimal(str(exp_setup["integration_time"])).as_tuple().exponent
        )

        num_possible_units = int(
            (ref_header_dict["time_to_run"] - _time_offset) // exp_setup[offset_var]
        )
        # Calculate start_times and round off correctly
        start_times = [
            round(exp_setup[offset_var] * i + _time_offset, abs(_precision))
            for i in range(num_possible_units)
        ]
    elif "start_times" in exp_setup:
        num_possible_units = len(exp_setup["start_times"])
        start_times = exp_setup["start_times"]

    return start_times, num_possible_units


def run_pert_processes(args: dict, local_exp_setup: dict, processes: list):
    """Run a list of perturbation processes, save experiment info and possibly
    compress the data dir if in ERDA mode.

    Parameters
    ----------
    args : dict
        Local run-time arguments
    local_exp_setup : dict
        Local experiment setup
    processes : list
        The processes to run
    """

    if len(processes) > 0:
        pt_runner.main_run(
            processes,
            args=args,
            n_units=args["n_units"],
        )
        # Save exp setup to exp folder
        g_save.save_exp_info(local_exp_setup, args)

        if args["erda_run"]:
            path = pl.Path(args["datapath"], local_exp_setup["folder_name"])
            g_save_utils.compress_dir(path)
    else:
        print("No processes to run - check if blocks already exists")
