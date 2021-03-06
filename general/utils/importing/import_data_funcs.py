import sys

sys.path.append("..")
import itertools as it
import json
import pathlib as pl
import re
import math
import colorama as col
import config as cfg
import general.utils.custom_decorators as dec
import general.utils.exceptions as g_exceptions
import general.utils.util_funcs as g_utils
import general.utils.running.runner_utils as r_utils
from libs.libutils import file_utils as lib_file_utils
import numpy as np
from general.params.model_licences import Models
from general.utils.module_import.type_import import *

if cfg.MODEL == Models.SHELL_MODEL:
    # Shell model specific imports
    import shell_model_experiments.utils.special_params as sh_sparams
    from shell_model_experiments.params.params import PAR, ParamsStructType

    # Get parameters for model
    params = PAR
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    # Get parameters for model
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams

    # Get parameters for model
    params = l63_params
    sparams = l63_sparams


@dec.check_dimension
def import_header(folder: Union[str, pl.Path] = "", file_name: str = "") -> dict:
    path = pl.Path(folder, file_name)

    # Import header
    header = ""
    header_size = 1

    with open(path, "r") as file:
        for i in range(header_size):
            header += (
                file.readline().rstrip().lstrip().strip("#").strip().replace(" ", "")
            )
        # Split only on "," if not inside []
        header = re.split(r",(?![^\[]*\])", header)

    header_dict = {}
    for item in header:
        splitted_item = item.split("=")

        if splitted_item[0] == "f":
            header_dict[splitted_item[0]] = np.complex(splitted_item[1])
        elif splitted_item[0] == "forcing":
            header_dict[splitted_item[0]] = np.complex(splitted_item[1])
        elif splitted_item[1] == "None":
            header_dict[splitted_item[0]] = None
        else:
            try:
                header_dict[splitted_item[0]] = float(splitted_item[1])
            except:
                header_dict[splitted_item[0]] = splitted_item[1]

    return header_dict


def import_info_file(path):
    """Import an info file from a given dir

    Parameters
    ----------
    path : str, Path
        The path to the dir containing the info file

    Returns
    -------
    dict
        The info
    """

    # Import reference header dict
    info_file_path = list(pl.Path(path).glob("*.txt"))
    if len(info_file_path) > 1:
        raise ValueError(
            f"To many data info files. Found " + f"{len(info_file_path)}; expected 1."
        )
    elif len(info_file_path) == 0:
        raise ValueError(f"No data info files found in the dir {path}")

    info_dict = import_header(file_name=info_file_path[0])

    return info_dict


def imported_sorted_perturbation_info(folder_name, args, search_pattern="*.csv"):
    """Import sorted lists of perturbation info

    Both time positions, legend strings, header dicts and file names are imported

    Parameters
    ----------
    folder_name : str, Path
        The sub folder to args["datapath"] in which to search for perturbation info
    args : dict
        Run-time arguments
    search_pattern : str
        Pattern with which perturbation files are searched

    Returns
    -------
    tuple
        Tuple with all perturbation info
    """
    # Initiate lists
    path_to_search = pl.Path(args["datapath"], folder_name)
    perturb_file_names = list(path_to_search.glob(search_pattern))
    perturb_time_pos_list_legend = []
    perturb_time_pos_list = []
    perturb_header_dicts = []

    if len(perturb_file_names) == 0:
        raise ImportError(
            f"No perturb files to import in the following dir: {str(path_to_search)}"
        )

    # Import headers and extract info
    for perturb_file in perturb_file_names:
        # Import perturbation header info
        perturb_header_dict = import_header(file_name=perturb_file)

        # Take submodel into account (if ATL -> time goes backwards)
        _temp_time_pos = int(round(perturb_header_dict["perturb_pos"]))
        if "submodel" in perturb_header_dict:
            if perturb_header_dict["submodel"] is not None:
                if perturb_header_dict["submodel"] == "ATL":
                    # Subtract number of datapoints in order to be able to match
                    # ATL perturbation against reference file import
                    _temp_time_pos = _temp_time_pos - int(
                        round(
                            perturb_header_dict["time_to_run"]
                            * perturb_header_dict["sample_rate"]
                        )
                    )
                    if _temp_time_pos < 0:
                        raise ValueError(
                            f"Time position becomes negative due to submodel={perturb_header_dict['submodel']}"
                        )

        perturb_time_pos_list.append(_temp_time_pos)

        # Save time positions, by taking into account different sample rates
        perturb_time_pos_list_legend.append(
            f"Start time: "
            + f'{perturb_header_dict["perturb_pos"]/perturb_header_dict["sample_rate"]*perturb_header_dict["dt"]:.3f}s'
        )

        perturb_header_dicts.append(perturb_header_dict)

    # Sort according to profile and run_in_profile if the keys are present
    if "profile" in perturb_header_dicts[0]:
        (
            perturb_file_names,
            ascending_sort_index,
        ) = g_utils.sort_paths_according_to_header_dicts(
            perturb_file_names, ["profile", "run_in_profile"]
        )
    else:
        # Get sort index
        ascending_sort_index = np.argsort(perturb_time_pos_list)
        # Sort perturb file names
        perturb_file_names = [perturb_file_names[i] for i in ascending_sort_index]

    # Sort arrays/lists
    perturb_time_pos_list = np.array(
        [perturb_time_pos_list[i] for i in ascending_sort_index]
    )
    perturb_time_pos_list_legend = np.array(
        [perturb_time_pos_list_legend[i] for i in ascending_sort_index]
    )
    perturb_header_dicts = [perturb_header_dicts[i] for i in ascending_sort_index]

    _offset = args["file_offset"] if "file_offset" in args else 0
    # Truncate at n_files and file_offset
    if args["n_files"] < np.inf:
        perturb_file_names = perturb_file_names[_offset : args["n_files"] + _offset]
        perturb_time_pos_list = perturb_time_pos_list[
            _offset : args["n_files"] + _offset
        ]
        perturb_time_pos_list_legend = perturb_time_pos_list_legend[
            _offset : args["n_files"] + _offset
        ]
        perturb_header_dicts = perturb_header_dicts[_offset : args["n_files"] + _offset]
    else:
        perturb_file_names = perturb_file_names[_offset:]
        perturb_time_pos_list = perturb_time_pos_list[_offset:]
        perturb_time_pos_list_legend = perturb_time_pos_list_legend[_offset:]
        perturb_header_dicts = perturb_header_dicts[_offset:]

    return (
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        perturb_header_dicts,
        perturb_file_names,
    )


def import_data(
    file_name: pl.Path,
    start_line: int = 0,
    max_lines: int = None,
    step: int = 1,
    dtype=None,
) -> Tuple[np.ndarray, dict]:

    # Import header
    header_dict = import_header(file_name=file_name)

    # Test length of file
    # with open(file_name) as file:
    #     for i, _ in enumerate(file):
    #         pass
    #     len_file = i  # Not +1 to skip header from length

    # if len_file == 0:
    #     raise ImportError(f"File is empty; file_name = {file_name}")

    # Set dtype if not given as argument
    if dtype is None:
        dtype = sparams.dtype

    stop_line = None if max_lines is None else start_line + max_lines
    # Import data
    with open(file_name) as file:
        # Setup iterator
        line_iterator = it.islice(
            file,
            start_line,
            stop_line,
            step,
        )
        data_in: np.ndarray(dtype=dtype) = np.genfromtxt(
            line_iterator, dtype=dtype, delimiter=","
        )

    if data_in.size == 0:
        raise ImportError(
            "No data was imported; file contained empty lines. "
            + f"File_name = {file_name}"
        )

    if len(data_in.shape) == 1:
        data_in = np.reshape(data_in, (1, data_in.size))

    return data_in, header_dict


def import_ref_data(args: dict = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Import reference file consisting of multiple records"""

    ref_record_names = list(pl.Path(args["datapath"], "ref_data").glob("*.csv"))
    if len(ref_record_names) == 0:
        raise ImportError("No reference csv files found")

    ref_files_sort_index = np.argsort(
        [str(ref_record_name) for ref_record_name in ref_record_names]
    )
    ref_record_files_sorted = [ref_record_names[i] for i in ref_files_sort_index]

    time_concat = []
    u_data_concat = []

    if len(args["specific_ref_records"]) == 1 and args["specific_ref_records"][0] < 0:
        records_to_import = ref_files_sort_index
        print("\n Importing all reference data records\n")
    elif args["ref_end_time"] > 0:
        # Calculate how many ref records to import
        start_rec_number = math.floor(
            args["ref_start_time"] / cfg.GLOBAL_PARAMS.record_max_time
        )
        end_rec_number = math.ceil(
            args["ref_end_time"] / cfg.GLOBAL_PARAMS.record_max_time
        )

        num_records = end_rec_number - start_rec_number
        records_to_import = [i + start_rec_number for i in range(num_records)]

        print(
            "\n Importing all/subset of data in listed reference data records: ",
            records_to_import,
            "\n",
        )
    else:
        records_to_import = args["specific_ref_records"]
        print("\n Importing listed reference data records: ", records_to_import, "\n")

    for i, record in enumerate(records_to_import):
        file_name = ref_record_files_sorted[record]

        endpoint = args["endpoint"] if "endpoint" in args else False

        # Only use ref_start_time information when importing first ref record
        start_line = (
            int(
                (args["ref_start_time"] % cfg.GLOBAL_PARAMS.record_max_time)
                * params.tts
            )
            if i == 0
            else 0
        )

        # If only importing one reference file, estimate max_line from ref_end_time
        # and ref_start_time
        if len(records_to_import) == 1:
            max_line = int(
                round(
                    (args["ref_end_time"] - args["ref_start_time"]) * params.tts
                    + (args["ref_start_time"] == 0)
                    + endpoint,
                    0,
                )
            )
        else:
            # Only use ref_end_time information if importing last record out of multiple
            remaining_time = args["ref_end_time"] % cfg.GLOBAL_PARAMS.record_max_time
            max_line = (
                int(
                    round(
                        remaining_time * params.tts
                        + (args["ref_start_time"] == 0)
                        + endpoint,
                        0,
                    )
                )
                if i + 1 == len(records_to_import) and remaining_time > 0
                else None
            )

        data_in, header_dict = import_data(
            file_name,
            start_line=start_line,
            max_lines=max_line
            if args["ref_end_time"] > args["ref_start_time"]
            else None,
        )
        # Add time offset according to the ref_id
        time_concat.append(data_in[:, 0])
        u_data_concat.append(data_in[:, 1:])

        # Add offset to first record, if not starting with first rec_id
        if i == 0:
            time_concat[0] += (
                round(header_dict["time_to_run"] / header_dict["n_records"])
                * header_dict["rec_id"]
            )

    # Add offset to time arrays to make one linear increasing time series
    for i, time_series in enumerate(time_concat):
        if i == 0:
            continue
        time_series += time_concat[i - 1][-1]
    time = np.concatenate(time_concat)
    u_data = np.concatenate(u_data_concat)

    ref_header_dict = import_info_file(pl.Path(args["datapath"], "ref_data"))

    return time, u_data, ref_header_dict


def import_perturbation_velocities(
    args: dict = None, search_pattern: str = "*.csv", raw_perturbations: bool = True
):
    """Import of perturbation velocities

    Parameters
    ----------
    args : dict, optional
        Run-time arguments, by default None
    search_pattern : str, optional
        Pattern to search for datafiles, by default "*.csv"
    raw_perturbations : bool, optional
        Toggles if the raw perturbation (ref data subtracted) or the
        perturbation rel. reference should be returned, by default True

    Returns
    -------
    tuple
        Collection of variables given by:
        (
            list: u_stores,
            list: perturb_time_pos_list,
            list: perturb_time_pos_list_legend,
            list: returned_perturb_header_dicts,
            list: u_ref_stores,
        )

    Raises
    ------
    g_exceptions.InvalidRuntimeArgument
        Raised if shell_cutoff is not None when running with a model other than
        shell model
    ValueError
        Raised if no datapath is specified
    """

    if cfg.MODEL != Models.SHELL_MODEL:
        if "shell_cutoff" in args:
            if args["shellcutoff"] is not None:
                raise g_exceptions.InvalidRuntimeArgument(argument="shell_cutoff")

    u_stores = []
    u_ref_stores = []
    returned_perturb_header_dicts = []

    if args["datapath"] is None:
        raise ValueError("No path specified")

    # Check if ref path exists
    ref_header_dict = import_info_file(pl.Path(args["datapath"], "ref_data"))

    (
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        perturb_header_dicts,
        perturb_file_names,
    ) = imported_sorted_perturbation_info(
        args["exp_folder"],
        args,
        search_pattern=search_pattern,
    )

    # Match the positions to the relevant ref files
    ref_file_match = g_utils.match_start_positions_to_ref_file(
        ref_header_dict=ref_header_dict, positions=perturb_time_pos_list
    )
    ref_file_match_keys_array = np.array(list(ref_file_match.keys()))

    # Get sorted file paths
    ref_record_files_sorted = lib_file_utils.get_files_in_path(
        pl.Path(args["datapath"], "ref_data")
    )

    ref_file_counter = 0
    perturb_index = 0

    for iperturb_file, perturb_file_name in enumerate(perturb_file_names):

        sum_pert_files = sum(
            [
                ref_file_match[ref_file_index].size
                for ref_file_index in ref_file_match_keys_array[
                    : (ref_file_counter + 1)
                ]
            ]
        )
        # If starting on importing from the next ref file
        if iperturb_file + 1 > sum_pert_files:
            # Bump counter
            ref_file_counter += 1
            # Reset perturbation index
            perturb_index = 0

        # Limit the number of perturbation files to import
        if iperturb_file < args["file_offset"]:
            perturb_index += 1
            continue

        perturb_data_in, perturb_header_dict = import_data(perturb_file_name)
        returned_perturb_header_dicts.append(perturb_header_dict)

        # Initialise ref_data_in of null size
        ref_data_in = np.array([], dtype=sparams.dtype).reshape(0, params.sdim + 1)

        # Determine offset to work with perturb_pos=0
        # perturb_time_pos_list is used in order to take into account adjusted
        # position when import ATL perturbation
        _offset = 1 * (perturb_time_pos_list[iperturb_file] == 0)

        # Keep importing datafiles untill ref_data_in has same size as perturb dataarray
        counter = 0
        while ref_data_in.shape[0] < perturb_header_dict["N_data"]:
            skip_lines = (
                ref_file_match[ref_file_match_keys_array[ref_file_counter]][
                    perturb_index
                ]
                + _offset
                if counter == 0
                else 0
            )
            max_rows = (
                int(perturb_header_dict["N_data"])
                if counter == 0
                else int(perturb_header_dict["N_data"]) - ref_data_in.shape[0] + 1
            )
            temp_ref_data_in, ref_header_dict = import_data(
                ref_record_files_sorted[
                    ref_file_match_keys_array[ref_file_counter] + counter
                ],
                start_line=skip_lines,
                max_lines=max_rows,
            )
            ref_data_in = (
                np.concatenate((ref_data_in, temp_ref_data_in))
                if temp_ref_data_in.size > 0
                else ref_data_in
            )
            counter += 1

        # If raw_perturbations is False, the reference data is not subtracted from perturbation
        if "shell_cutoff" in args:
            if args["shell_cutoff"] is not None:
                # Calculate error array
                # Add +2 to args["shell_cutoff"] since first entry is time, and args["shell_cutoff"]
                # starts from 0
                u_stores.append(
                    perturb_data_in[:, 1 : (args["shell_cutoff"] + 2)]
                    - (raw_perturbations)
                    * ref_data_in[:, 1 : (args["shell_cutoff"] + 2)]
                )
            else:
                # Calculate error array
                u_stores.append(
                    perturb_data_in[:, 1:] - (raw_perturbations) * ref_data_in[:, 1:]
                )
        else:
            # Calculate error array
            u_stores.append(
                perturb_data_in[:, 1:] - (raw_perturbations) * ref_data_in[:, 1:]
            )

        # If perturb positions are the same for all perturbations, return
        # ref_data_in too
        # if np.unique(perturb_time_pos_list).size == 1:
        u_ref_stores.append(ref_data_in[:, 1:])
        # else:
        #     u_ref_stores = None

        if args["n_files"] is not None:
            if iperturb_file + 1 - args["file_offset"] >= args["n_files"]:
                break

        perturb_index += 1

    return (
        u_stores,
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        returned_perturb_header_dicts,
        u_ref_stores,
    )


def import_start_u_profiles(
    args: dict = None, start_times: list = []
) -> Tuple[np.ndarray, List[int], dict]:
    """Import all u profiles to start perturbations from

    Parameters
    ----------
    args : dict, optional
        Run-time arguments, by default None

    Returns
    -------
    tuple
        (
            u_init_profiles : np.ndarray((params.sdim + 2 * params.bd_size, n_profiles * n_runs_per_profile))
                The initial velocity profiles
            positions : The index position of the profiles
            ref_header_dict : The header dictionary of the reference data file
        )

    Raises
    ------
    KeyError
        Raised if an unknown key is used to store the time_to_run information
        in the reference header
    """
    len_start_times = len(start_times)
    if args["start_times"] is not None and len_start_times > 0:
        print(
            f"{col.Fore.RED}In import_start_u_profiles: Both args['start_times'] and kwarg start_times specified. start_times overrides args['start_times']{col.Fore.RESET}"
        )

    n_profiles = args["n_profiles"] if len_start_times == 0 else len_start_times
    n_runs_per_profile = args["n_runs_per_profile"]  # if len_start_times == 0 else 1

    # Check if ref path exists
    ref_file_path = pl.Path(args["datapath"], "ref_data")
    # Get ref info text file
    ref_header_dict = import_info_file(ref_file_path)

    # If start_times not specified -> prepare random start times
    if args["start_times"] is None and len_start_times == 0:
        r_utils.get_random_start_times(args, n_profiles, ref_header_dict)
        _temp_start_times = args["start_times"]
    elif len_start_times > 0:
        _temp_start_times = start_times
    else:
        _temp_start_times = args["start_times"]

    print(
        f"\nImporting {n_profiles} velocity profiles positioned as "
        + "requested in reference datafile\n"
    )
    # Make sure to round and convert to int in a proper way
    positions = np.round(np.array(_temp_start_times) * params.tts).astype(np.int64)

    print(
        "\nPositions of perturbation start: ",
        positions * params.stt,
        "(in seconds)",
    )

    # Match the positions to the relevant ref files
    ref_file_match = g_utils.match_start_positions_to_ref_file(
        ref_header_dict=ref_header_dict, positions=positions
    )

    # Get sorted file paths
    ref_record_files_sorted = lib_file_utils.get_files_in_path(
        pl.Path(args["datapath"], "ref_data")
    )

    # Prepare u_init_profiles matrix
    u_init_profiles = np.zeros(
        (params.sdim + 2 * params.bd_size, n_profiles * n_runs_per_profile),
        dtype=sparams.dtype,
    )

    # Import velocity profiles
    _counter = 0

    for file_id in ref_file_match.keys():
        for position in ref_file_match[int(file_id)]:
            temp_u_init_profile = np.genfromtxt(
                ref_record_files_sorted[int(file_id)],
                dtype=sparams.dtype,
                delimiter=",",
                skip_header=np.int64(round(position, 0)),
                max_rows=1,
            )

            # Skip time datapoint and pad array with zeros
            if n_runs_per_profile == 1:
                indices = _counter
                u_init_profiles[sparams.u_slice, indices] = temp_u_init_profile[1:]

                # Update counter
                _counter += 1
            elif n_runs_per_profile > 1:
                indices = np.s_[_counter : _counter + n_runs_per_profile : 1]
                u_init_profiles[sparams.u_slice, indices] = np.repeat(
                    np.reshape(
                        temp_u_init_profile[1:], (temp_u_init_profile[1:].size, 1)
                    ),
                    n_runs_per_profile,
                    axis=1,
                )

                # Update counter
                _counter += n_runs_per_profile

    return (
        u_init_profiles,
        positions,
        ref_header_dict,
    )


def import_exp_info_file(args: dict):
    """Import the experiment info file

    Parameters
    ----------
    args : dict
        Run-time arguments

    Returns
    -------
    dict
        The experiment info file parsed as a dict

    Raises
    ------
    ImportError
        Raised if no subfolder is given to search for the experiment info file
    ValueError
        Raised if more than one experiment info file are found
    ImportError
        Raised if no experiment info file is found.
    """

    subfolder = pl.Path("")
    # Add pert_vector_folder if present
    if "pert_vector_folder" in args:
        if args["pert_vector_folder"] is not None:
            subfolder /= args["pert_vector_folder"]

    # Add the specified exp_folder
    if args["exp_folder"] is not None:
        subfolder /= args["exp_folder"]
    else:
        raise ImportError("No valid subfolder to search for exp_setup")

    path = pl.Path(args["datapath"], subfolder)

    json_files = list(path.glob("*.json"))
    len_files = len(json_files)

    if len_files > 1:
        raise ValueError(
            f"To many experiment info files. Found " + f"{len_files}; expected 1."
        )
    elif len_files == 0:
        raise ImportError(f"No experiment info file found at path {path}")

    # Get experiment setup
    with open(json_files[0], "r") as file:
        exp_info_file = json.load(file)

    return exp_info_file
