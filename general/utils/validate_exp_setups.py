def validate_lorentz_block_setup(exp_setup={}):
    # Check if enough time is set off to the run
    if exp_setup["time_to_run"] < exp_setup["n_analyses"] * exp_setup["day_offset"]:
        raise ValueError(
            "Exp. setup invalid: The time_to_run is less than required. "
            + "Please ensure that it fits the day_offset times the number of analyses"
            + " as a minimum"
        )


def validate_start_time_method(exp_setup={}):
    """Check that the setup is consistent in terms of the method to determine
    the start times"""

    if "block_offset" in exp_setup and "start_times" in exp_setup:
        raise ValueError(
            "Exp. setup invalid: Both block_offset and start_times entries ARE"
            + " set in the experiment setup. This is not valid; choose one of them."
        )
    elif "block_offset" not in exp_setup and "start_times" not in exp_setup:
        raise ValueError(
            "Exp. setup invalid: Both block_offset and start_times entries are"
            + " NOT set in the experiment setup. This is not valid; choose one of them."
        )
