from general.params.experiment_licences import Experiments as EXP
import general.utils.exceptions as g_exceptions
import config as cfg


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

    if cfg.LICENCE == EXP.LORENTZ_BLOCK:
        offset_var = "block_offset"
    elif (
        cfg.LICENCE == EXP.BREEDING_VECTORS
        or cfg.LICENCE == EXP.LYAPUNOV_VECTORS
        or cfg.LICENCE == EXP.SINGULAR_VECTORS
    ):
        offset_var = "vector_offset"
    else:
        raise g_exceptions.LicenceImplementationError(licence=cfg.LICENCE)

    if "start_times" in exp_setup and "eval_times" in exp_setup:
        raise g_exceptions.ExperimentSetupError(
            "Both 'start_times' and 'eval_times' present in experiment setup"
        )

    if offset_var in exp_setup and "start_times" in exp_setup:
        if len(exp_setup["start_times"]) != 1:
            raise g_exceptions.ExperimentSetupError(
                f"Exp. setup invalid: Both {offset_var} and start_times (more than one) entries ARE"
                + " set in the experiment setup. This is not valid; choose one of them to govern start times."
            )
    elif offset_var in exp_setup and "eval_times" in exp_setup:
        if len(exp_setup["eval_times"]) != 1:
            raise g_exceptions.ExperimentSetupError(
                f"Exp. setup invalid: Both {offset_var} and eval_times (more than one) entries ARE"
                + " set in the experiment setup. This is not valid; choose one of them to govern start times."
            )
        else:
            if (
                cfg.LICENCE == EXP.BREEDING_VECTORS
                or cfg.LICENCE == EXP.LYAPUNOV_VECTORS
            ):
                if (
                    exp_setup["eval_times"][0]
                    - exp_setup["integration_time"] * exp_setup["n_cycles"]
                    < 0
                ):
                    raise g_exceptions.ExperimentSetupError(
                        "Too long integration time, or too many cycles, compared"
                        + "to the chosen evaluation time",
                        exp_variable=f"eval_time = {exp_setup['eval_times'][0]};"
                        + f" integration_time = {exp_setup['integration_time']}",
                    )
            elif cfg.LICENCE == EXP.SINGULAR_VECTORS:
                if exp_setup["eval_times"][0] - exp_setup["integration_time"] < 0:
                    raise g_exceptions.ExperimentSetupError(
                        "Too long integration time compared to the chosen evaluation time",
                        exp_variable=f"eval_time = {exp_setup['eval_times'][0]};"
                        + f" integration_time = {exp_setup['integration_time']}",
                    )

    elif offset_var not in exp_setup and "start_times" not in exp_setup:
        raise g_exceptions.ExperimentSetupError(
            f"Exp. setup invalid: Both {offset_var}, start_times and eval_times entries are"
            + " NOT set in the experiment setup. This is not valid; choose one of them."
        )
