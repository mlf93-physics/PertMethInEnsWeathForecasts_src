import os
import pathlib as pl
import random
import itertools as it
import numpy as np
import general.utils.util_funcs as g_utils
from libs.libutils import file_utils as lib_file_utils
import general.utils.importing.import_data_funcs as g_import
import general.utils.exceptions as g_exceptions
from general.utils.module_import.type_import import *
from general.params.model_licences import Models
import config as cfg

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    from shell_model_experiments.params.params import ParamsStructType
    from shell_model_experiments.params.params import PAR as PAR_SH
    import shell_model_experiments.utils.special_params as sh_sparams

    params = PAR_SH
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams

    params = l63_params
    sparams = l63_sparams


def import_lorentz_block_perturbations(args=None, raw_perturbations=True):
    """Imports perturbations from perturbation dir stored as lorentz perturbation
    and match them up with reference data. Returns a lorentz block list which
    contains perturbations rel. reference data.

    Parameters
    ----------
    raw_perturbations : bool
        If perturbations are returned relative to the reference or as absolute
        perturbations

    """

    if cfg.MODEL != Models.SHELL_MODEL:
        if "shell_cutoff" in args:
            if args["shell_cutoff"] is not None:
                raise g_exceptions.InvalidRuntimeArgument(argument="shell_cutoff")

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
        ref_header_dict=ref_header_dict, positions=perturb_time_pos_list
    )

    # Get sorted file paths
    ref_record_files_sorted = lib_file_utils.get_files_in_path(
        pl.Path(args["datapath"], "ref_data")
    )

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
            ref_record_files_sorted[ref_file_match_keys_array[ref_file_counter]],
            start_line=start_index,
            max_lines=perturb_offset * (num_blocks),
            step=perturb_offset,
        )

        # If raw_perturbations is False, the reference data is not subtracted
        # from perturbation
        if "shell_cutoff" in args:
            if args["shell_cutoff"] is not None:
                # Calculate error array Add +2 to args["shell_cutoff"] since
                # first entry is time, and args["shell_cutoff"] starts from 0
                lorentz_block_stores.append(
                    perturb_data_in[:, 1 : (args["shell_cutoff"] + 2)]
                    - (raw_perturbations)
                    * ref_data_in[:, 1 : (args["shell_cutoff"] + 2)]
                )
                lorentz_block_ref_stores.append(
                    ref_data_in[:, 1 : (args["shell_cutoff"] + 2)]
                )

            else:
                # Calculate error array if shell_cutoff is None
                lorentz_block_stores.append(
                    perturb_data_in[:, 1:] - (raw_perturbations) * ref_data_in[:, 1:]
                )
                lorentz_block_ref_stores.append(ref_data_in[:, 1:])

        else:
            # Calculate error array if shell_cutoff is not in args
            lorentz_block_stores.append(
                perturb_data_in[:, 1:] - (raw_perturbations) * ref_data_in[:, 1:]
            )
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


def import_profiles_for_nm_analysis(args: dict = None) -> Tuple[np.ndarray, dict]:
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
    ref_record_files_sorted = lib_file_utils.get_files_in_path(
        pl.Path(args["datapath"], "ref_data")
    )

    ref_header_dict = g_import.import_info_file(pl.Path(args["datapath"], "ref_data"))

    num_ref_records = len(ref_record_files_sorted)
    profiles = []

    for ifile, ref_file in enumerate(ref_record_files_sorted):
        with open(ref_file) as file:
            lines = random.sample(
                list(it.chain(file)),
                num_profiles // num_ref_records
                + (ifile + 1 == num_ref_records) * num_profiles % num_ref_records,
            )

        lines = map(lambda x: x.strip().split(","), lines)
        profiles.extend(list(lines))

    profiles = np.array(profiles, dtype=sparams.dtype)[:, 1:]

    # Pad profiles if necessary
    if params.bd_size > 0:
        profiles = np.pad(
            profiles,
            pad_width=((0, 0), (params.bd_size, params.bd_size)),
            mode="constant",
        )

    profiles = profiles.T

    return profiles, ref_header_dict


def import_perturb_vectors(
    args: dict,
    raw_perturbations: bool = False,
    dtype=None,
    force_no_ref_import: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[dict]]:
    """Import units of perturbation vectors, e.g. BVs or SVs

    Parameters
    ----------
    args : dict
        Run-time arguments
    raw_perturbations : bool
        If relevant u_profiles should be subtracted the vectors (False) or not (True)
    dtype
        The datatype of the vectors to import
    force_no_ref_import : bool
        (True) Force function to not import start_u_profiles and return them.
        KEEP IN MIND THAT THIS ALTERS THE NUMBER OF STATED RETURN VARIABLES

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, list, list]
        (
            vector_units : np.ndarray((n_units, n_vectors, sdim))
                The perturbation vectors stored as units.
            characteristic_values : np.ndarray((n_units, n_vectors))
                The characteristic values for the vectors, e.g. singular values
            u_init_profiles : np.ndarray((n_units, n_vectors, sdim))
                The ref velocity profiles at evaluation time
            eval_pos : List
                The index position of the evaluation time
            perturb_header_dicts : List
                List of header dicts of the vector units
        )

    Raises
    ------
    ImportError
        Raised if the pert_vector_folder is not found on disc
    g_exceptions.InvalidRuntimeArgument
        Raised if the function is unable to infer the number of vector files to
        import from args
    """

    # Infer the number of files
    if "n_units" in args:
        if args["n_units"] < np.inf:
            args["n_files"] = args["n_units"]
        elif "n_profiles" in args:
            args["n_files"] = args["n_profiles"]
        else:
            raise g_exceptions.InvalidRuntimeArgument(
                "Unable to infer the number of files from args"
            )
    elif "n_profiles" in args:
        args["n_files"] = args["n_profiles"]
    else:
        raise g_exceptions.InvalidRuntimeArgument(
            "Unable to infer the number of files from args"
        )

    # Get experiment info
    exp_info = g_import.import_exp_info_file(args)

    # Infer number of runs per profile from exp info file
    if args["n_runs_per_profile"] < 0:
        args["n_runs_per_profile"] = exp_info["n_vectors"]

    if not os.path.isdir(pl.Path(args["datapath"], args["pert_vector_folder"])):
        raise ImportError(f"No perturbation path named {args['pert_vector_folder']}")

    (
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        perturb_header_dicts,
        perturb_file_names,
    ) = g_import.imported_sorted_perturbation_info(
        pl.Path(args["pert_vector_folder"], args["exp_folder"]),
        args,
        search_pattern="*vectors*.csv",
    )

    # Prepare start_time to import correct u_profiles
    eval_pos: List[int] = []
    for i in range(len(perturb_file_names)):
        eval_pos.append(int(round(perturb_header_dicts[i]["val_pos"])))

    args["start_times"] = np.array(eval_pos) * params.stt

    # Import reference data
    if not force_no_ref_import:
        (
            u_init_profiles,
            _,
            _,
        ) = g_import.import_start_u_profiles(args=args)
    else:
        u_init_profiles = None

    vector_units = np.empty(
        (args["n_files"], args["n_runs_per_profile"], params.sdim), dtype=np.complex128
    )

    characteristic_values = np.zeros(
        (args["n_files"], args["n_runs_per_profile"]), dtype=np.complex128
    )

    for i, file_name in enumerate(perturb_file_names):
        vector_unit, _ = g_import.import_data(
            file_name,
            max_lines=args["n_runs_per_profile"],
            dtype=dtype,
            start_line=args["specific_start_vector"] + 1,
        )

        # Skip characteristic value if present
        if vector_unit.shape[1] == params.sdim + 1:
            characteristic_values[i, :] = vector_unit[:, 0]
            vector_units[i, :, :] = vector_unit[:, 1:]
        else:
            vector_units[i, :, :] = vector_unit

        if i + 1 >= args["n_files"]:
            break

    # Take real part if in Lorentz model - only zeros present in imag part
    if cfg.MODEL == Models.LORENTZ63:
        vector_units = vector_units.real

    if not raw_perturbations:
        # u_init_profiles is reshaped to fit shape (n_units, n_runs_per_profile, sdim)
        # of vector_units array
        vector_units = vector_units - np.reshape(
            u_init_profiles[sparams.u_slice, :].T,
            (args["n_files"], args["n_runs_per_profile"], params.sdim),
        )

    return (
        vector_units,
        characteristic_values,
        u_init_profiles,
        eval_pos,
        perturb_header_dicts,
    )
