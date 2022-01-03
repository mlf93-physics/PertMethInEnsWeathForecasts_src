import sys

sys.path.append("../../")
import re
import pathlib as pl
from collections import OrderedDict

import config as cfg
import general.utils.importing.import_data_funcs as g_import
import numpy as np
from general.params.model_licences import Models
from general.utils.module_import.type_import import *
from libs.libutils import file_utils as lib_file_utils

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    from shell_model_experiments.params.params import ParamsStructType
    from shell_model_experiments.params.params import PAR as PAR_SH

    params = PAR_SH
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params

    params = l63_params


def match_start_positions_to_ref_file(
    ref_header_dict: dict = None, positions: List[int] = None
) -> OrderedDict:
    """Match the positions to their respective record of the reference
    dataseries."""

    ref_file_match = OrderedDict()

    model_dt = ref_header_dict["dt"]

    # The file ids which shall be imported
    matched_file_ids = positions // int(
        ref_header_dict["record_max_time"] * ref_header_dict["sample_rate"] / model_dt
    )

    # The positions in the files that shall be imported
    positions_relative_to_record = positions % int(
        ref_header_dict["record_max_time"] * ref_header_dict["sample_rate"] / model_dt
    )

    # Save the positions per record id to a dictionary
    for file_id in set(matched_file_ids):
        record_positions = np.where(matched_file_ids == file_id)[0]
        ref_file_match[file_id] = positions_relative_to_record[record_positions]

    return ref_file_match


def adjust_start_times_with_offset(args):

    if args["start_times"] is not None:
        if args["n_profiles"] > 1 and args["start_time_offset"] is None:
            np.testing.assert_equal(
                len(args["start_times"]),
                args["n_profiles"],
                "The number of start times do not equal the number of"
                + " requested profiles.",
            )
        elif args["n_profiles"] > 1 and args["start_time_offset"] is not None:
            np.testing.assert_equal(
                len(args["start_times"]), 1, "Too many start times given"
            )
            print(
                "Determining starttimes from single starttime value and the"
                + " start_time_offset parameter"
            )
            args["start_times"] = [
                args["start_times"][0] + args["start_time_offset"] * i
                for i in range(args["n_profiles"])
            ]
        else:
            np.testing.assert_equal(
                len(args["start_times"]), 1, "Too many start times given"
            )

    return args


def get_exp_files_and_names(args, type: str = "perturbations") -> list:
    """Get the experiment data file names

    Parameters
    ----------
    args : dict
        Run-time arguments
    type : str, optional
        The type of the experiment files, by default "perturbations"
        Possible types: perturbations, vectors

    Returns
    -------
    list
        The names of the experiment files
    """

    if type == "perturbations":
        path = pl.Path(args["datapath"], args["exp_folder"])
    elif type == "vectors":
        path = pl.Path(args["datapath"], args["pert_vector_folder"], args["exp_folder"])

    names = lib_file_utils.get_file_names_in_path(path)
    files = lib_file_utils.get_files_in_path(path)

    return names, files


def get_header_dicts_from_paths(file_paths: List[pl.Path]) -> List[dict]:
    """Get list of header dicts from path

    Parameters
    ----------
    file_paths : List[pl.Path]
        The list of file paths

    Returns
    -------
    List[dict]
        The list of header dicts
    """
    # Get headers
    header_dicts: list = []
    for path in file_paths:
        header_dicts.append(g_import.import_header(path.parent, path.name))

    return header_dicts


def handle_different_headers(header_dict):
    """Handle new and old style of the header_dict

    Parameters
    ----------
    header_dict : dict
        The header of a datafile parsed as a dict
    """

    if "f" in header_dict:
        header_dict["forcing"] = header_dict["f"]
        del header_dict["f"]

    if "time" in header_dict:
        header_dict["time_to_run"] = header_dict["time"]
        del header_dict["time"]

    return header_dict


def determine_params_from_header_dict(header_dict: dict, args: dict):
    if cfg.MODEL == Models.SHELL_MODEL:

        # Save parameters to args dict:
        args["forcing"] = header_dict["forcing"].real
        args["ny_n"] = header_dict["ny_n"]
        args["ny"] = header_dict["ny"]
        args["diff_exponent"] = header_dict["diff_exponent"]

        # if args["ny_n"] is None:

        #     if args["forcing"] == 0:
        #         args["ny_n"] = 0
        #     else:
        #         args["ny_n"] = sh_utils.ny_n_from_ny_and_forcing(
        #             args["forcing"], header_dict["ny"], header_dict["diff_exponent"]
        #         )
        #     # Take ny from reference file
        # else:
        #     args["ny"] = sh_utils.ny_from_ny_n_and_forcing(
        #         args["forcing"], args["ny_n"], args["diff_exponent"]
        #     )

    elif cfg.MODEL == Models.LORENTZ63:
        print("Nothing specific to do with args in lorentz63 model yet")


def normalize_array(
    array: np.ndarray, norm_value: float = 1e-2, axis: int = 0
) -> np.ndarray:
    """Normalize an array of arbitrary dimension to a given value along a given
    axis

    Parameters
    ----------
    array : np.ndarray
        The array to be normalized
    norm_value : float, optional
        The specified norm of the array after normalization, by default 1e-2
    axis : int, optional
        The axis along which the normalization should be performed, by default 0

    Returns
    -------
    np.ndarray
        The normalized array
    """
    # Find scaling factor in order to have the seeked norm of the vector
    lambda_factor = norm_value / np.linalg.norm(array, axis=axis)

    # Add missing axis to make multiplication with array possible
    lambda_factor = np.expand_dims(lambda_factor, axis=axis)

    # Scale down the vector
    array = lambda_factor * array

    return array


def get_values_from_dicts(dicts: list, key: str) -> list:
    """Get the values from a list of dicts according to a specific key

    Parameters
    ----------
    dicts : list
        The list of dictionaries
    key : str
        The key from which the values are taken

    Returns
    -------
    list
        The list of values matching the key
    """

    value_list: list = []

    for i, dict in enumerate(dicts):
        if key not in dict:
            raise ValueError(f"No key '{key}' in dict")

        value_list.append(dict[key])

    value_list = value_list

    return value_list


def sort_paths_according_to_header_dicts(
    paths: List[pl.Path], keys: List[str]
) -> Tuple[List[pl.Path], List[int]]:
    """Sort a list of paths according to the specified keys in the header_dict
    of the file at the path

    Parameters
    ----------
    paths : List[pl.Path]
        The list of paths to sort
    keys : List[str]
        A list of keys in the header_dicts to sort according to

    Returns
    -------
    Tuple[List[pl.Path], List[int]]
        (
            list:
                List of sorted paths
            list:
                List of the index to sort another array in the same way as the paths.
        )
    """

    # Get headers
    header_dicts: list = []
    for path in paths:
        header_dicts.append(g_import.import_header(path.parent, path.name))

    value_lists: list = []
    for key in keys:
        value_lists.append(get_values_from_dicts(header_dicts, key))

    # Sort paths
    sorted_parths = []
    sort_index = []

    for _, _, enum_path in sorted(zip(*value_lists, enumerate(paths))):
        sorted_parths.append(enum_path[1])
        sort_index.append(enum_path[0])

    return sorted_parths, sort_index
