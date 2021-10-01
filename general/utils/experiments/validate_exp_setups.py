from general.params.experiment_licences import Experiments as EXP
import general.utils.exceptions as g_exceptions
from config import LICENCE


def validate_lorentz_block_setup(exp_setup={}):
    # Check if enough time is set off to the run
    if exp_setup["time_to_run"] < exp_setup["n_analyses"] * exp_setup["day_offset"]:
        raise ValueError(
            "Exp. setup invalid: The time_to_run is less than required. "
            + "Please ensure that it fits the day_offset times the number of analyses"
            + " as a minimum"
        )


def validate_start_time_method(exp_setup: dict = {}):
    """Check that the setup is consistent in terms of the method to determine
    the start times"""

    if LICENCE == EXP.LORENTZ_BLOCK:
        offset_var = "block_offset"
    elif LICENCE == EXP.BREEDING_VECTORS:
        offset_var = "vector_offset"
    else:
        raise g_exceptions.LicenceImplementationError(licence=LICENCE)

    if offset_var in exp_setup and "start_times" in exp_setup:
        if len(exp_setup["start_times"]) > 1:
            raise ValueError(
                f"Exp. setup invalid: Both {offset_var} and start_times (more than one) entries ARE"
                + " set in the experiment setup. This is not valid; choose one of them to govern start times."
            )
    elif offset_var not in exp_setup and "start_times" not in exp_setup:
        raise ValueError(
            f"Exp. setup invalid: Both {offset_var} and start_times entries are"
            + " NOT set in the experiment setup. This is not valid; choose one of them."
        )
