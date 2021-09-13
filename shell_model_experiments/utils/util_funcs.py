import os
import pathlib as pl
from collections import OrderedDict
import numpy as np
from pathlib import Path
from shell_model_experiments.params.params import *


def match_start_positions_to_ref_file(args=None, header_dict=None, positions=None):
    """Match the positions to their respective record of the reference
    dataseries."""

    ref_file_match = OrderedDict()

    # The file ids which shall be imported
    matched_file_ids = positions // int(
        header_dict["record_max_time"] * header_dict["sample_rate"] / dt
    )

    # The positions in the files that shall be imported
    positions_relative_to_record = positions % int(
        header_dict["record_max_time"] * header_dict["sample_rate"] / dt
    )

    # Save the positions per record id to a dictionary
    for file_id in set(matched_file_ids):
        record_positions = np.where(matched_file_ids == file_id)[0]
        ref_file_match[file_id] = positions_relative_to_record[record_positions]

    return ref_file_match


def get_sorted_ref_record_names(args=None):
    # Get file paths
    ref_record_names = list(Path(args["path"], "ref_data").glob("*.csv"))
    ref_files_sort_index = np.argsort(
        [str(ref_record_name) for ref_record_name in ref_record_names]
    )
    ref_record_names_sorted = [ref_record_names[i] for i in ref_files_sort_index]

    return ref_record_names_sorted


def adjust_start_times_with_offset(args):

    if args["start_time"] is not None:
        if args["n_profiles"] > 1 and args["start_time_offset"] is None:
            np.testing.assert_equal(
                len(args["start_time"]),
                args["n_profiles"],
                "The number of start times do not equal the number of"
                + " requested profiles.",
            )
        elif args["n_profiles"] > 1 and args["start_time_offset"] is not None:
            np.testing.assert_equal(
                len(args["start_time"]), 1, "Too many start times given"
            )
            print(
                "Determining starttimes from single starttime value and the"
                + " start_time_offset parameter"
            )
            args["start_time"] = [
                args["start_time"][0] + args["start_time_offset"] * i
                for i in range(args["n_profiles"])
            ]
        else:
            np.testing.assert_equal(
                len(args["start_time"]), 1, "Too many start times given"
            )

    return args


def count_existing_files_or_dirs(search_path="", search_pattern="*.csv"):
    """Counts the number of files or folder of a specific kind, and returns the
    the count. To be used to store numbered folders/files in a numbered order."""

    search_path = pl.Path(search_path)

    # Check if path exists
    seach_path_exists = os.path.isdir(search_path)
    if seach_path_exists:
        if search_pattern != "/":
            n_files = len(list(search_path.glob(search_pattern)))
        else:
            dirs = list(search_path.glob("*"))
            dirs = [dirs[i] for i in range(len(dirs)) if os.path.isdir(dirs[i])]
            n_files = len(dirs)
    else:
        n_files = 0

    return n_files
