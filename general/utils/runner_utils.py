import sys

sys.path.append("..")
import pathlib as pl
import general.utils.importing.import_data_funcs as g_import
from general.params.experiment_licences import Experiments as EXP
import general.utils.exceptions as g_exceptions
from config import LICENCE


def generate_start_times(exp_setup, args):
    ref_header_dict = g_import.import_info_file(pl.Path(args["datapath"], "ref_data"))

    if LICENCE == EXP.LORENTZ_BLOCK:
        offset_var = "block_offset"
    elif LICENCE == EXP.BREEDING_VECTORS or LICENCE == EXP.LYAPUNOV_VECTORS:
        offset_var = "vector_offset"
    else:
        raise g_exceptions.LicenceImplementationError(licence=LICENCE)

    if offset_var in exp_setup:
        if "start_times" in exp_setup:
            _time_offset = exp_setup["start_times"][0]
        elif "eval_times" in exp_setup:
            if LICENCE == EXP.BREEDING_VECTORS:
                _time_offset = (
                    exp_setup["eval_times"][0]
                    - exp_setup["n_cycles"] * exp_setup["integration_time"]
                )
            elif LICENCE == EXP.LYAPUNOV_VECTORS:
                _time_offset = (
                    exp_setup["eval_times"][0] - exp_setup["integration_time"]
                )
        else:
            _time_offset = 0

        num_possible_units = int(
            (ref_header_dict["time_to_run"] - _time_offset) // exp_setup[offset_var]
        )
        start_times = [
            exp_setup[offset_var] * i + _time_offset for i in range(num_possible_units)
        ]
    elif "start_times" in exp_setup:
        num_possible_units = len(exp_setup["start_times"])
        start_times = exp_setup["start_times"]

    return start_times, num_possible_units
