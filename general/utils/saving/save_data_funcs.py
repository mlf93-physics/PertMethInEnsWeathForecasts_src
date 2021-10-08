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
from config import MODEL, GLOBAL_PARAMS


def save_data(data_out, subsubfolder="", prefix="", perturb_position=None, args=None):
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

    if GLOBAL_PARAMS.ref_run:
        subsubfolder = "ref_data"
        # Generate path if not existing
        expected_path = g_save_utils.generate_dir(
            pl.Path("data", stand_data_name, subsubfolder), args=args
        )

        prefix = "ref_"

        ref_filename_extra = f"_rec{args['record_id']}"
        ref_data_info_name = f"{expected_path}/ref_data_info_{stand_data_name}.txt"

        info_line = g_save_utils.args_to_string(args)

        if MODEL == Models.SHELL_MODEL:
            info_line += (
                f", n_f={sh_params.n_forcing}, dt={sh_params.dt}, "
                + f"epsilon={sh_params.epsilon}, lambda={sh_params.lambda_const}, "
                + f"sample_rate={sh_params.sample_rate}"
            )
        elif MODEL == Models.LORENTZ63:
            info_line += f", dt={l63_params.dt}, sample_rate={l63_params.sample_rate}"

        with open(ref_data_info_name, "w") as file:
            file.write(info_line)

        ref_header_extra = f", rec_id={args['record_id']}"
        header = g_save_utils.generate_header(
            args, n_data=n_data, append_extra=ref_header_extra
        )

    else:
        ref_filename_extra = ""
        # Generate path if not existing
        expected_path = g_save_utils.generate_dir(
            pl.Path(args["datapath"], args["exp_folder"], subsubfolder), args=args
        )

        if perturb_position is not None:
            perturb_header_extra = f", perturb_pos={int(perturb_position)}"
            header = g_save_utils.generate_header(
                args, n_data=n_data, append_extra=perturb_header_extra
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


def save_perturb_info(args=None, exp_setup=None):
    """Save info textfile about the perturbation runs

    Parameters
    ----------
    args : dict
        A dictionary containing run-time arguments

    """

    expected_path = g_save_utils.generate_dir(
        pl.Path(args["datapath"], args["exp_folder"]), args=args
    )
    # Prepare filename
    perturb_data_info_name = pl.Path(expected_path)
    stand_data_name = g_save_utils.generate_standard_data_name(args)
    perturb_data_info_name /= f"perturb_data_info_{stand_data_name}.txt"

    # Check if path already exists
    dir_exists = os.path.isfile(perturb_data_info_name)
    if dir_exists:
        print("Perturb info not saved, since file already exists")
        return

    print("Saving perturb data info textfile\n")

    # Prepare line to write
    info_line_args = g_save_utils.args_to_string(args)
    exp_setup_line = g_save_utils.args_to_string(exp_setup)

    if MODEL == Models.SHELL_MODEL:
        info_line = (
            f"n_f={sh_params.n_forcing}, dt={sh_params.dt}, "
            + f"epsilon={sh_params.epsilon}, lambda={sh_params.lambda_const}, "
            + f"sample_rate={sh_params.sample_rate}, "
        )
    elif MODEL == Models.LORENTZ63:
        info_line = f"sample_rate={sh_params.sample_rate}, dt={sh_params.dt}, "

    append_extra = f", experiment={config.LICENCE}, "

    info_line += info_line_args + append_extra + exp_setup_line

    # Write to file
    with open(str(perturb_data_info_name), "w") as file:
        file.write(info_line)


def save_exp_info(exp_info, args):

    # Generate standard name
    stand_data_name = g_save_utils.generate_standard_data_name(args)

    # Generate out file name
    out_name = f"_{stand_data_name}"
    prefix = "exp_info"

    # Generate path if not existing
    expected_path = g_save.g_save_utils.generate_dir(
        pl.Path(args["datapath"], args["exp_folder"]), args=args
    )

    out_path = pl.Path(expected_path, f"{prefix}{out_name}.json")

    with open(out_path, "w") as file:
        json.dump(exp_info, file)


if __name__ == "__main__":
    dir = "./data/ny2.37e-08_t4.00e+02_n_f0_f1.0/lorentz_block_short_pred_ttr0.25/"
    zip_name = "test_tar"

    g_save_utils.compress_dir(dir, zip_name)
