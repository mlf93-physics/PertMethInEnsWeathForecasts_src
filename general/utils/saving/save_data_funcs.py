import os
import sys

sys.path.append("..")
import json
import pathlib as pl
import numpy as np
import shell_model_experiments.params as sh_params
import config
import lorentz63_experiments.params.params as l63_params
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_utils as g_save_utils
from general.params.model_licences import Models
import config as cfg

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    params = sh_params
elif cfg.MODEL == Models.LORENTZ63:
    params = l63_params


def save_data(
    data_out: np.ndarray,
    subsubfolder: str = "",
    prefix: str = "",
    perturb_position: int = None,
    args: dict = None,
) -> pl.Path:
    """Save the data to disc.

    Parameters
    ----------
    data_out : array
        Data to save
    subfolder : str
        Optional subfolder to put the data in
    prefix : str
        Prefix to the datafiles
    perturb_position : int
        The perturbation position
    args : dict
        A dictionary containing run-time arguments

    """

    if args is None:
        print("Please supply an argument dictionary to save data.")
        exit()

    # Prepare variables to be used when saving
    n_data = data_out.shape[0]
    stand_data_name = g_save_utils.generate_standard_data_name(args)

    if cfg.GLOBAL_PARAMS.ref_run:
        # Generate path if not existing
        expected_path = g_save_utils.generate_dir(
            pl.Path("data", stand_data_name, "ref_data"), args=args
        )

        prefix = "ref_"
        ref_filename_extra = f"_rec{args['record_id']}"
        ref_header_extra = f"rec_id={args['record_id']}, "
        header = g_save_utils.generate_header(
            args, n_data=n_data, append_extra=ref_header_extra
        )

    else:
        ref_filename_extra = ""
        # Generate path if not existing
        expected_path = g_save_utils.generate_dir(
            pl.Path(args["datapath"], args["out_exp_folder"], subsubfolder), args=args
        )

        # Prepare extra header items
        perturb_header_extra = ""
        if perturb_position is not None:
            perturb_header_extra = f"perturb_pos={perturb_position}, "

        header = g_save_utils.generate_header(
            args,
            n_data=n_data,
            append_extra=perturb_header_extra,
            append_options=["licence"],
        )

    # Generate out file name
    out_name = f"udata_{stand_data_name}"

    # Save data
    np.savetxt(
        f"{expected_path}/{prefix}{out_name}{ref_filename_extra}.csv",
        data_out,
        delimiter=",",
        header=header,
    )

    return expected_path


def save_reference_info(args):

    print("Prepare ref data info textfile\n")

    stand_data_name = g_save_utils.generate_standard_data_name(args)

    # Generate path if not existing
    expected_path = g_save_utils.generate_dir(
        pl.Path("data", stand_data_name, "ref_data"), args=args
    )

    ref_data_info_path = f"{expected_path}/ref_data_info_{stand_data_name}.txt"

    info_line = g_save_utils.args_to_string(args)
    append_extra = f"record_max_time={cfg.GLOBAL_PARAMS.record_max_time}, "
    info_line += g_save_utils.generate_header(
        args, args["Nt"] * params.sample_rate, append_extra=append_extra
    )

    # Write to file
    save_run_info(ref_data_info_path, info_line)


def save_perturb_info(args=None, exp_setup=None):
    """Save info textfile about the perturbation runs

    Parameters
    ----------
    args : dict
        A dictionary containing run-time arguments

    """

    expected_path = g_save_utils.generate_dir(
        pl.Path(args["datapath"], args["out_exp_folder"]), args=args
    )
    # Prepare filename
    perturb_data_info_path = pl.Path(expected_path)
    stand_data_name = g_save_utils.generate_standard_data_name(args)
    perturb_data_info_path /= f"perturb_data_info_{stand_data_name}.txt"

    print("Prepare perturb data info textfile\n")

    # Prepare line to write
    exp_setup_line = g_save_utils.args_to_string(exp_setup)

    info_line = g_save_utils.generate_header(
        args, args["Nt"] * params.sample_rate, append_options=["licence"]
    )
    info_line += exp_setup_line

    # Write to file
    save_run_info(perturb_data_info_path, info_line)


def save_run_info(info_path: pl.Path, info_line: str):

    # Check if path already exists
    dir_exists = os.path.isfile(info_path)
    if dir_exists:
        print(f"Info file not saved, since file already exists at path {info_path}")
        return

    print("Saving run info file\n")

    # Write to file
    with open(str(info_path), "w") as file:
        file.write(info_line)


def save_exp_info(exp_info: dict, args: dict):
    """Save the experiment info to a json file in the exp_folder

    Parameters
    ----------
    exp_info : dict
        The experiment info
    args : dict
        Run-time arguments
    """

    # Generate standard name
    stand_data_name = g_save_utils.generate_standard_data_name(args)

    # Generate out file name
    out_name = f"_{stand_data_name}"
    prefix = "exp_info"

    # Generate path if not existing
    expected_path = g_save.g_save_utils.generate_dir(
        pl.Path(args["datapath"], args["out_exp_folder"]), args=args
    )

    out_path = pl.Path(expected_path, f"{prefix}{out_name}.json")

    with open(out_path, "w") as file:
        json.dump(exp_info, file)

    print("\nExperiment info saved to file\n")


if __name__ == "__main__":
    dir = "./data/ny2.37e-08_t4.00e+02_n_f0_f1.0/lorentz_block_short_pred_ttr0.25/"
    zip_name = "test_tar"

    g_save_utils.compress_dir(dir, zip_name)
