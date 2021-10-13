import os
import pathlib as pl
import random
import itertools as it
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

    if args["datapath"] is None:
        raise ValueError("No path specified")

    # Check if ref path exists
    ref_file_path = pl.Path(args["datapath"], "ref_data")

    # Get ref info text file
    ref_header_path = list(pl.Path(ref_file_path).glob("*.txt"))[0]
    # Import header info
    ref_header_dict = g_import.import_header(file_name=ref_header_path)

    (
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        perturb_header_dicts,
        perturb_file_names,
    ) = g_import.imported_sorted_perturbation_info(args["exp_folder"], args)
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


def import_profiles_for_nm_analysis(args=None):
    """Import random positioned u profiles for normal mode analysis

    Parameters
    ----------
    args : dict, optional
        Run-time arguments, by default None

    Returns
    -------
    tuple
        profiles, ref_header_dict : The imported profiles and the reference header dict
    """

    n_profiles = args["n_profiles"]
    n_runs_per_profile = args["n_runs_per_profile"]
    num_profiles = n_profiles * n_runs_per_profile

    # Get sorted file paths
    ref_record_names_sorted = g_utils.get_sorted_ref_record_names(args=args)

    ref_header_dict = g_import.import_info_file(pl.Path(args["datapath"], "ref_data"))

    num_ref_records = len(ref_record_names_sorted)
    profiles = []

    for ifile, ref_file in enumerate(ref_record_names_sorted):
        with open(ref_file) as file:
            lines = random.sample(
                list(it.chain(file)),
                num_profiles // num_ref_records
                + (ifile + 1 == num_ref_records) * num_profiles % num_ref_records,
            )

        lines = map(lambda x: x.strip().split(","), lines)
        profiles.extend(list(lines))

    profiles = np.array(profiles, dtype=l63_params.dtype)[:, 1:]

    # Pad profiles if necessary
    if params.bd_size > 0:
        profiles = np.pad(
            profiles,
            pad_width=((0, 0), (params.bd_size, params.bd_size)),
            mode="constant",
        )

    profiles = profiles.T

    return profiles, ref_header_dict


def import_perturb_vectors(args):
    # Set arguments
    args["n_files"] = args["n_units"]

    pt_vector_dirname = "pt_vectors"

    if not os.path.isdir(pl.Path(args["datapath"], pt_vector_dirname)):
        raise ImportError(f"No perturbation path named {pt_vector_dirname}")

    (
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        perturb_header_dicts,
        perturb_file_names,
    ) = g_import.imported_sorted_perturbation_info(
        pl.Path(pt_vector_dirname, args["exp_folder"]),
        args,
        search_pattern="*vectors*.csv",
    )

    vector_units = []

    for i, file_name in enumerate(perturb_file_names):
        vector_unit, _ = g_import.import_data(
            file_name, max_lines=args["n_runs_per_profile"] + 1
        )

        # Prepare start_time to import correct u_profiles
        args["start_times"] = [perturb_header_dicts[i]["val_pos"] * params.stt]

        # Import reference data
        (
            u_init_profiles,
            _,
            _,
        ) = g_import.import_start_u_profiles(args=args)

        vector_units.append(vector_unit - u_init_profiles.T)
        if i + 1 >= args["n_units"]:
            break

    vector_units = np.array(vector_units)

    return vector_units, perturb_header_dicts
