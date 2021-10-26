import pathlib as pl
import json


def get_exp_setup(path_to_file: pl.Path, args: dict) -> dict:
    """Get experiment setup from json file

    Parameters
    ----------
    path_to_file : str, Path
        Path to the experiment setup file (.json file)
    args : dict
        Run-time arguments

    Returns
    -------
    dict
        Experiment setup

    Raises
    ------
    ValueError
        If no exp_setup is given to the run-time argument dict
    """
    # Get experiment setup
    with open(path_to_file, "r") as file:
        exp_setup_file = json.load(file)

    if args["exp_setup"] is None:
        raise ValueError("No experiment setup chosen")
    else:
        exp_setup = exp_setup_file[args["exp_setup"]]

    return exp_setup
