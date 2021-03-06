import sys

sys.path.append("../../")
import re
import pathlib as pl
import json
from operator import itemgetter
import numpy as np
from libs.libutils import file_utils as lib_file_utils, type_utils as lib_type_utils


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


def preprocess_exp_setup_for_comparison(exp_setup: dict) -> None:
    """Preprocess the experiment setup to be used in comparison runners

    Parameters
    ----------
    exp_setup : dict
        The experiment setup
    """

    if "unit_offset" in exp_setup["general"]:
        # Add unit_offset to vector gen exp setups
        exp_setup["bv_gen_setup"]["vector_offset"] = exp_setup["general"]["unit_offset"]
        exp_setup["sv_gen_setup"]["vector_offset"] = exp_setup["general"]["unit_offset"]

    # Add sub_exp_folder to vector gen exp setups
    exp_setup["bv_gen_setup"]["sub_exp_folder"] = exp_setup["general"]["sub_exp_folder"]
    exp_setup["sv_gen_setup"]["sub_exp_folder"] = exp_setup["general"]["sub_exp_folder"]

    if "lv_gen_setup" in exp_setup:
        if "unit_offset" in exp_setup["general"]:
            exp_setup["lv_gen_setup"]["vector_offset"] = exp_setup["general"][
                "unit_offset"
            ]
        exp_setup["lv_gen_setup"]["sub_exp_folder"] = exp_setup["general"][
            "sub_exp_folder"
        ]

    if "fsv_gen_setup" in exp_setup:
        if "unit_offset" in exp_setup["general"]:
            exp_setup["fsv_gen_setup"]["vector_offset"] = exp_setup["general"][
                "unit_offset"
            ]
        exp_setup["fsv_gen_setup"]["sub_exp_folder"] = exp_setup["general"][
            "sub_exp_folder"
        ]


def update_compare_exp_folders(args, specific_runs_per_profile_dict=None):
    """Update the exp_folders argument when comparison folder is given as exp_folder
    argument

    Parameters
    ----------
    args : dict
        Run-time arguments
    """

    if args["exp_folder"] is not None:
        # Get dirs in path
        _path = pl.Path(args["datapath"], args["exp_folder"])
        _dirs = lib_file_utils.get_dirs_in_path(_path, recursively=True)

        len_folders = len(_dirs)

        if len_folders == 0:
            args["exp_folders"] = [args["exp_folder"]]
        else:
            # Sort out dirs not named according to input arguments
            _exp_folders: list = []
            for item in args["perturbations"]:
                # Update specific_runs according to dict
                if specific_runs_per_profile_dict is not None:
                    args["specific_runs_per_profile"] = specific_runs_per_profile_dict[
                        item
                    ]

                temp_new_folders = [
                    pl.Path(args["exp_folder"], _dirs[i].name)
                    for i in range(len_folders)
                    if re.match(
                        fr"{item}(\d+_perturbations|_perturbations)", _dirs[i].name
                    )
                ]
                # Filter out unwanted folders
                if (
                    args["specific_runs_per_profile"] is not None
                    and len(temp_new_folders) > 1
                ):
                    temp_new_folders = [
                        temp_new_folders[i]
                        for i in range(len(temp_new_folders))
                        if lib_type_utils.get_digits_from_string(
                            temp_new_folders[i].name
                        )
                        in args["specific_runs_per_profile"]
                    ]
                else:
                    temp_new_folders = temp_new_folders[: args["n_runs_per_profile"]]

                # Add new folders to list
                _exp_folders.extend([str(folder) for folder in temp_new_folders])
            for item in args["vectors"]:
                _exp_folders.extend(
                    [
                        str(pl.Path(args["exp_folder"], *_dirs[i].parts[-2:]))
                        for i in range(len_folders)
                        if re.match(fr"{item}(\d+_vectors|_vectors)", _dirs[i].name)
                    ]
                )

            args["exp_folders"] = sorted(_exp_folders)
