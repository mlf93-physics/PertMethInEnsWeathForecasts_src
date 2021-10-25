import sys

sys.path.append("..")
import itertools as it
import re
import json
import pathlib as pl
import numpy as np
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
from general.params.model_licences import Models
import general.utils.util_funcs as g_utils
import general.utils.exceptions as g_exceptions
from config import MODEL

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params


def import_header(folder="", file_name=None):
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

        # Save time positions, by taking into account different sample rates
        perturb_time_pos_list.append(int(perturb_header_dict["perturb_pos"]))

        perturb_time_pos_list_legend.append(
            f"Start time: "
            + f'{perturb_header_dict["perturb_pos"]/perturb_header_dict["sample_rate"]*perturb_header_dict["dt"]:.3f}s'
        )

        perturb_header_dicts.append(perturb_header_dict)

    # Get sort index
    ascending_perturb_pos_index = np.argsort(perturb_time_pos_list)

    # Sort arrays/lists
    perturb_time_pos_list = np.array(
        [
            perturb_time_pos_list[i]
            for i in ascending_perturb_pos_index
            if (i + 1) <= args["n_files"]
        ]
    )
    perturb_time_pos_list_legend = np.array(
        [
            perturb_time_pos_list_legend[i]
            for i in ascending_perturb_pos_index
            if (i + 1) <= args["n_files"]
        ]
    )
    perturb_header_dicts = [
        perturb_header_dicts[i]
        for i in ascending_perturb_pos_index
        if (i + 1) <= args["n_files"]
    ]
    perturb_file_names = [
        perturb_file_names[i]
        for i in ascending_perturb_pos_index
        if (i + 1) <= args["n_files"]
    ]

    return (
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        perturb_header_dicts,
        perturb_file_names,
    )


def import_data(file_name, start_line=0, max_lines=None, step=1):

    # Import header
    header_dict = import_header(file_name=file_name)

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
        data_in: np.ndarray(dtype=params.dtype) = np.genfromtxt(
            line_iterator, dtype=params.dtype, delimiter=","
        )

    if len(data_in.shape) == 1:
        data_in = np.reshape(data_in, (1, data_in.size))

    return data_in, header_dict


def import_ref_data(args=None):
    """Import reference file consisting of multiple records"""

    ref_record_names = list(pl.Path(args["datapath"], "ref_data").glob("*.csv"))
    ref_files_sort_index = np.argsort(
        [str(ref_record_name) for ref_record_name in ref_record_names]
    )
    ref_record_names_sorted = [ref_record_names[i] for i in ref_files_sort_index]

    time_concat = []
    u_data_concat = []

    if len(args["specific_ref_records"]) == 1 and args["specific_ref_records"][0] < 0:
        records_to_import = ref_files_sort_index
        print("\n Importing all reference data records\n")
    else:
        records_to_import = args["specific_ref_records"]
        print("\n Importing listed reference data records: ", records_to_import, "\n")

    for i, record in enumerate(records_to_import):
        file_name = ref_record_names_sorted[record]

        endpoint = args["endpoint"] if "endpoint" in args else False

        data_in, header_dict = import_data(
            file_name,
            start_line=int(args["ref_start_time"] * params.tts),
            max_lines=int(
                round(
                    (args["ref_end_time"] - args["ref_start_time"]) * params.tts
                    + (args["ref_start_time"] == 0)
                    + endpoint,
                    0,
                )
            )
            if args["ref_end_time"] > args["ref_start_time"]
            else None,
        )
        # Add time offset according to the ref_id
        time_concat.append(data_in[:, 0])
        u_data_concat.append(data_in[:, 1:])

        # Add offset to first record, if not starting with first rec_id
        if i == 0:
            time_concat[0] += data_in.shape[0] / params.tts * header_dict["rec_id"]

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

    if MODEL != Models.SHELL_MODEL:
        if "shell_cutoff" in args:
            if args["shellcutoff"] is not None:
                raise g_exceptions.InvalidRuntimeArgument(argument="shell_cutoff")

    u_stores = []
    returned_perturb_header_dicts = []

    if args["datapath"] is None:
        raise ValueError("No path specified")

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

    # Check if ref path exists
    ref_header_dict = import_info_file(pl.Path(args["datapath"], "ref_data"))
    # Match the positions to the relevant ref files
    ref_file_match = g_utils.match_start_positions_to_ref_file(
        args=args, ref_header_dict=ref_header_dict, positions=perturb_time_pos_list
    )
    ref_file_match_keys_array = np.array(list(ref_file_match.keys()))

    # Get sorted file paths
    ref_record_names_sorted = g_utils.get_sorted_ref_record_names(args=args)

    ref_file_counter = 0
    perturb_index = 0

    for iperturb_file, perturb_file_name in enumerate(perturb_file_names):

        sum_pert_files = sum(
            [
                len(ref_file_match[ref_file_index])
                for ref_file_index in ref_file_match_keys_array[
                    : (ref_file_counter + 1)
                ]
            ]
        )

        if iperturb_file + 1 > sum_pert_files:
            ref_file_counter += 1
            perturb_index = 0

        if iperturb_file < args["file_offset"]:
            perturb_index += 1
            continue

        perturb_data_in, perturb_header_dict = import_data(perturb_file_name)
        returned_perturb_header_dicts.append(perturb_header_dict)

        # Initialise ref_data_in of null size
        ref_data_in = np.array([], dtype=params.dtype).reshape(0, params.sdim + 1)

        # Determine offset to work with perturb_pos=0
        _offset = 1 * (perturb_header_dict["perturb_pos"] == 0)

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
                else int(perturb_header_dict["N_data"]) - ref_data_in.shape[0]
            )

            temp_ref_data_in, ref_header_dict = import_data(
                ref_record_names_sorted[
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
        if np.unique(perturb_time_pos_list).size == 1:
            u_ref_stores = [ref_data_in[:, 1:]]
        else:
            u_ref_stores = None

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


def import_start_u_profiles(args=None):
    """Import all u profiles to start perturbations from"""

    n_profiles = args["n_profiles"]
    n_runs_per_profile = args["n_runs_per_profile"]

    # Check if ref path exists
    ref_file_path = pl.Path(args["datapath"], "ref_data")

    # Get ref info text file
    ref_header_dict = import_info_file(ref_file_path)

    if args["start_times"] is None:
        print(
            f"\nImporting {n_profiles} velocity profiles randomly positioned "
            + "in reference datafile(s)\n"
        )
        if "time" in ref_header_dict:
            time_to_run = ref_header_dict["time"]
        elif "time_to_run" in ref_header_dict:
            time_to_run = ref_header_dict["time_to_run"]
        else:
            raise KeyError("No valid key with time to run information")

        n_data = int((time_to_run) * params.tts)

        # Generate random start positions
        # division = total #datapoints - burn_in #datapoints - #datapoints per perturbation
        division_size = int(n_data // n_profiles - args["Nt"] * params.sample_rate)
        rand_division_start = np.random.randint(
            low=0, high=division_size, size=n_profiles
        )
        positions = np.array(
            [
                (division_size + args["Nt"] * params.sample_rate) * i
                + rand_division_start[i]
                for i in range(n_profiles)
            ]
        )
    else:
        print(
            f"\nImporting {n_profiles} velocity profiles positioned as "
            + "requested in reference datafile\n"
        )
        positions = np.array(args["start_times"]) * params.tts

    print(
        "\nPositions of perturbation start: ",
        positions * params.stt,
        "(in seconds)",
    )

    # Match the positions to the relevant ref files
    ref_file_match = g_utils.match_start_positions_to_ref_file(
        args=args, ref_header_dict=ref_header_dict, positions=positions
    )

    # Get sorted file paths
    ref_record_names_sorted = g_utils.get_sorted_ref_record_names(args=args)

    # Prepare u_init_profiles matrix
    u_init_profiles = np.zeros(
        (params.sdim + 2 * params.bd_size, n_profiles * n_runs_per_profile),
        dtype=params.dtype,
    )

    # Import velocity profiles
    counter = 0

    for file_id in ref_file_match.keys():
        for position in ref_file_match[int(file_id)]:
            temp_u_init_profile = np.genfromtxt(
                ref_record_names_sorted[int(file_id)],
                dtype=params.dtype,
                delimiter=",",
                skip_header=np.int64(round(position, 0)),
                max_rows=1,
            )

            # Skip time datapoint and pad array with zeros
            if n_runs_per_profile == 1:
                indices = counter
                u_init_profiles[params.u_slice, indices] = temp_u_init_profile[1:]
            elif n_runs_per_profile > 1:
                indices = np.s_[counter : counter + n_runs_per_profile : 1]
                u_init_profiles[params.u_slice, indices] = np.repeat(
                    np.reshape(
                        temp_u_init_profile[1:], (temp_u_init_profile[1:].size, 1)
                    ),
                    n_runs_per_profile,
                    axis=1,
                )

            counter += 1

    return (
        u_init_profiles,
        positions,
        ref_header_dict,
    )


def import_exp_info_file(args):

    if args["exp_folder"] is not None:
        subfolder = args["exp_folder"]
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
