from pathlib import Path
import numpy as np
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
import general.utils.util_funcs as g_utils
import general.utils.importing.import_data_funcs as g_import
from general.params.model_licences import Models
from config import MODEL

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params


def import_lorentz_block_perturbations(args=None, rel_ref=True):
    """Imports perturbations from perturbation dir stored as lorentz perturbation
    and match them up with reference data. Returns a lorentz block list which
    contains perturbations rel. reference data.

    Parameters
    ----------
    rel_ref : bool
        If perturbations are returned relative to the reference or as absolute
        perturbations

    """

    lorentz_block_stores = []
    lorentz_block_ref_stores = []

    if args["path"] is None:
        raise ValueError("No path specified")

    # Check if ref path exists
    ref_file_path = Path(args["path"], "ref_data")

    # Get ref info text file
    ref_header_path = list(Path(ref_file_path).glob("*.txt"))[0]
    # Import header info
    ref_header_dict = g_import.import_header(file_name=ref_header_path)

    (
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        perturb_header_dicts,
        perturb_file_names,
    ) = g_import.imported_sorted_perturbation_info(args)

    # Match the positions to the relevant ref files
    ref_file_match = g_utils.match_start_positions_to_ref_file(
        args=args, header_dict=ref_header_dict, positions=perturb_time_pos_list
    )

    # Get sorted file paths
    ref_record_names_sorted = g_utils.get_sorted_ref_record_names(args=args)

    ref_file_counter = 0
    perturb_index = 0

    for iperturb_file, perturb_file_name in enumerate(perturb_file_names):

        ref_file_match_keys_array = np.array(list(ref_file_match.keys()))
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

        # Import perturbation data
        perturb_data_in, perturb_header_dict = g_import.import_data(perturb_file_name)

        # Prepare for reference import
        num_blocks = perturb_data_in.shape[0]
        perturb_offset = int(perturb_header_dict["start_time_offset"] * params.tts)
        start_index = (
            ref_file_match[ref_file_match_keys_array[ref_file_counter]][perturb_index]
            + perturb_offset
        ) + 1

        # Import reference u vectors
        ref_data_in, ref_header_dict = g_import.import_data(
            ref_record_names_sorted[ref_file_match_keys_array[ref_file_counter]],
            start_line=start_index,
            max_lines=perturb_offset * (num_blocks),
            step=perturb_offset,
        )
        # Calculate error array
        if rel_ref:
            lorentz_block_stores.append(perturb_data_in[:, 1:] - ref_data_in[:, 1:])
        else:
            # Calculate error array
            lorentz_block_stores.append(perturb_data_in[:, 1:])
            lorentz_block_ref_stores.append(ref_data_in[:, 1:])

        if args["n_files"] is not None:
            if iperturb_file + 1 - args["file_offset"] >= args["n_files"]:
                break

        perturb_index += 1

    return (
        lorentz_block_stores,
        lorentz_block_ref_stores,
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        perturb_header_dicts,
    )
