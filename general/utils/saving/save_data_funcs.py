import os
import sys

sys.path.append("..")
import json
import pathlib as pl
import subprocess as sp
import numpy as np
import shell_model_experiments.params as sh_params
import config
import lorentz63_experiments.params.params as l63_params
import general.utils.saving.save_data_funcs as g_save
from general.params.model_licences import Models
from config import MODEL, GLOBAL_PARAMS


def args_to_string(args):
    """Convert args dictionary to string

    Parameters
    ----------
    args : dict
        A dictionary containing run-time arguments

    """
    if args is None:
        return ""

    arg_str_list = [f"{key}={value}" for key, value in args.items()]
    arg_str = ", ".join(arg_str_list)

    return arg_str


def generate_dir(expected_path, subfolder="", args=None):
    """Generate a directory from a path and possibly a subfolder

    Parameters
    ----------
    expected_path : str, Path
        Path to the dir
    subfolder : str, optional
        Subfolder to append to the path
    args : dict
        A dictionary containing run-time arguments

    """

    if len(subfolder) == 0:
        # See if folder is present
        dir_exists = os.path.isdir(expected_path)

        if not dir_exists:
            os.makedirs(expected_path)

        subfolder = expected_path
    else:
        # Check if path exists
        expected_path = str(pl.Path(expected_path, subfolder))
        dir_exists = os.path.isdir(expected_path)

        if not dir_exists:
            os.makedirs(expected_path)

    return expected_path


def compress_dir(path_to_dir, zip_name):
    """Compress a directory using tar. The resulting .tar.gz file will be
    located at path_to_dir

    Parameters
    ----------
    path_to_dir : str, Path
        Path to the directory/file to compress
    zip_name : str
        The name of the produced zip

    """
    if not os.path.isdir(path_to_dir):
        raise ValueError(f"No dir at the given path ({path_to_dir})")
    else:
        path_to_dir = pl.Path(path_to_dir)

    out_name = pl.Path(path_to_dir.parent, zip_name + ".tar.gz")

    print(f"Compressing data directory at path: {path_to_dir}")
    sp.run(
        ["tar", "-czvf", str(out_name), "-C", path_to_dir.parent, path_to_dir.name],
        stdout=sp.DEVNULL,
    )


def generate_header(args, n_data=0, append_extra=""):
    """Generate a header string

    Parameters
    ----------
    args : dict
        A dictionary containing run-time arguments
    n_data : int
        Number of datapoints
    append_extra : str
        Some string to append to the header

    """

    header = args_to_string(args)

    if MODEL == Models.SHELL_MODEL:
        header += (
            f", n_f={sh_params.n_forcing}, dt={sh_params.dt}, epsilon={sh_params.epsilon}, "
            + f"lambda={sh_params.lambda_const}, N_data={n_data}, "
            + f"sample_rate={sh_params.sample_rate}, "
            + f"experiment={config.LICENCE}"
            + append_extra
        )
    elif MODEL == Models.LORENTZ63:
        header += (
            f", dt={l63_params.dt}, N_data={n_data}, "
            + f"sample_rate={l63_params.sample_rate}, "
            + f"experiment={config.LICENCE}"
            + append_extra
        )

    return header


def convert_arguments_to_string(args):
    """Convert specific arguments to string format using .format

    Parameters
    ----------
    args : dict
        A dictionary containing run-time arguments

    """

    arguments = {}

    # Some universal argument conversion
    arguments["time_to_run"] = "{:.2e}".format(args["time_to_run"])

    if MODEL == Models.SHELL_MODEL:
        arguments["forcing"] = "{:.1f}".format(args["forcing"])
        arguments["ny"] = "{:.2e}".format(args["ny"])
    elif MODEL == Models.LORENTZ63:
        arguments["sigma"] = "{:.2e}".format(args["sigma"])
        arguments["r_const"] = "{:.2e}".format(args["r_const"])
        arguments["b_const"] = "{:.2e}".format(args["b_const"])

    return arguments


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
    temp_args = convert_arguments_to_string(args)

    # expected_path = generate_dir(expected_name, subfolder=subfolder, args=args)

    if GLOBAL_PARAMS.ref_run:
        subsubfolder = "ref_data"
        # Generate path if not existing
        if MODEL == Models.SHELL_MODEL:
            expected_path = (
                f"data/ny{temp_args['ny']}_t{temp_args['time_to_run']}"
                + f"_n_f{sh_params.n_forcing}_f{temp_args['forcing']}"
            )
        elif MODEL == Models.LORENTZ63:
            expected_path = (
                f"data/sig{temp_args['sigma']}_t{temp_args['time_to_run']}"
                + f"_b{temp_args['b_const']}_r{temp_args['r_const']}_dt{l63_params.dt}"
            )

        expected_path = generate_dir(pl.Path(expected_path, subsubfolder), args=args)

        prefix = "ref_"

        ref_filename_extra = f"_rec{args['record_id']}"
        if MODEL == Models.SHELL_MODEL:
            ref_data_info_name = (
                f"{expected_path}/ref_data_info_ny"
                + f"{temp_args['ny']}_t{temp_args['time_to_run']}"
                + f"_n_f{sh_params.n_forcing}_f{temp_args['forcing']}.txt"
            )
        elif MODEL == Models.LORENTZ63:
            ref_data_info_name = (
                f"{expected_path}/ref_data_info_sig{temp_args['sigma']}"
                + f"_t{temp_args['time_to_run']}"
                + f"_b{temp_args['b_const']}_r{temp_args['r_const']}.txt"
            )

        arg_str_list = [f"{key}={value}" for key, value in args.items()]
        info_line = ", ".join(arg_str_list)

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
        header = generate_header(args, n_data=n_data, append_extra=ref_header_extra)

    else:
        ref_filename_extra = ""

        # Generate path if not existing
        expected_path = generate_dir(
            pl.Path(args["datapath"], args["exp_folder"], subsubfolder), args=args
        )

        if perturb_position is not None:
            perturb_header_extra = f", perturb_pos={int(perturb_position)}"
            header = generate_header(
                args, n_data=n_data, append_extra=perturb_header_extra
            )

    # Generate out file name
    if MODEL == Models.SHELL_MODEL:
        out_name = (
            f"udata_ny{temp_args['ny']}_t{temp_args['time_to_run']}"
            + f"_n_f{sh_params.n_forcing}_f{temp_args['forcing']}"
        )
    elif MODEL == Models.LORENTZ63:
        out_name = (
            f"udata_sig{temp_args['sigma']}"
            + f"_t{temp_args['time_to_run']}"
            + f"_b{temp_args['b_const']}_r{temp_args['r_const']}"
        )

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

    temp_args = convert_arguments_to_string(args)

    expected_path = generate_dir(
        pl.Path(args["datapath"], args["exp_folder"]), args=args
    )
    # Prepare filename
    perturb_data_info_name = pl.Path(expected_path)
    if MODEL == Models.SHELL_MODEL:
        perturb_data_info_name /= (
            f"perturb_data_info_ny{temp_args['ny']}_t{temp_args['time_to_run']}"
            + f"_n_f{sh_params.n_forcing}_f{temp_args['forcing']}.txt"
        )
    elif MODEL == Models.LORENTZ63:
        perturb_data_info_name /= (
            f"perturb_data_info_sig{temp_args['sigma']}"
            + f"_t{temp_args['time_to_run']}"
            + f"_b{temp_args['b_const']}_r{temp_args['r_const']}.txt"
        )

    # Check if path already exists
    dir_exists = os.path.isfile(perturb_data_info_name)
    if dir_exists:
        print("Perturb info not saved, since file already exists")
        return

    print("Saving perturb data info textfile\n")

    # Prepare line to write
    info_line_args = args_to_string(args)
    exp_setup_line = args_to_string(exp_setup)

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
    temp_args = g_save.convert_arguments_to_string(args)

    # Generate out file name
    if MODEL == Models.SHELL_MODEL:
        out_name = (
            f"_ny{temp_args['ny']}_t{temp_args['time_to_run']}"
            + f"_n_f{sh_params.n_forcing}_f{temp_args['forcing']}"
        )
    elif MODEL == Models.LORENTZ63:
        out_name = (
            f"_sig{temp_args['sigma']}"
            + f"_t{temp_args['time_to_run']}"
            + f"_b{temp_args['b_const']}_r{temp_args['r_const']}"
        )

    prefix = "exp_info"

    # Generate path if not existing
    expected_path = g_save.generate_dir(
        pl.Path(args["datapath"], args["exp_folder"]), args=args
    )

    out_path = pl.Path(expected_path, f"{prefix}{out_name}.json")

    with open(out_path, "w") as file:
        json.dump(exp_info, file)


if __name__ == "__main__":
    dir = "./data/ny2.37e-08_t4.00e+02_n_f0_f1.0/lorentz_block_short_pred_ttr0.25/"
    zip_name = "test_tar"

    compress_dir(dir, zip_name)
