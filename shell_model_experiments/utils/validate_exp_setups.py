def validate_lorentz_block_setup(exp_setup={}):
    # Check if enough time is set off to the run
    if exp_setup["time_to_run"] < exp_setup["n_analyses"] * exp_setup["day_offset"]:
        raise ValueError(
            "Exp. setup invalid: The time_to_run is less than required. "
            + "Please ensure that it fits the day_offset times the number of analyses"
            + " as a minimum"
        )
