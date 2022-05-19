from general.utils.module_import.type_import import *


def get_start_end_times_from_exp_setup(
    exp_setup: dict, pert_info_dict: dict
) -> Tuple[float, float]:
    """Prepare the start and end times from exp setup in order to import
    part of ref data

    Parameters
    ----------
    exp_setup : dict
        The experiment setup
    pert_info_dict : dict
        The perturbation info file dictionary

    Returns
    -------
    Tuple[float, float]
        (
            start_time : The start time for the reference import
            end_time : The end time for the reference import
        )

    Raises
    ------
    ValueError
        Raised if the start time could not be determined from exp_setup
    """
    if "time_to_run" in exp_setup:
        int_time = exp_setup["time_to_run"]
    elif "integration_time" in exp_setup:
        int_time = exp_setup["integration_time"]
    else:
        raise ValueError("No entry in exp_setup to get time duration of integration")

    # Prepare ref import
    if "start_times" in exp_setup:
        if pert_info_dict["save_last_pert"]:
            start_time = (
                exp_setup["start_times"][0] + (exp_setup["n_cycles"] - 1) * int_time
            )
            end_time = start_time + pert_info_dict["n_units"] * int_time
        else:
            start_time = exp_setup["start_times"][0]
            end_time = exp_setup["start_times"][0] + exp_setup["n_cycles"] * int_time
    elif "eval_times" in exp_setup:
        # Adjust start- and endtime differently depending on if only last
        # perturbation data is saved, or all perturbation data is saved.
        if pert_info_dict["save_last_pert"]:
            start_time = exp_setup["eval_times"][0]
            end_time = start_time + (pert_info_dict["n_units"]) * int_time
        else:
            start_time = exp_setup["eval_times"][0] - exp_setup["n_cycles"] * int_time
            end_time = exp_setup["eval_times"][0]
    else:
        raise ValueError("start_time could not be determined from exp setup")

    return start_time, end_time
