import os
from shell_model_experiments.config import LICENCE
import numpy as np
from pathlib import Path
from shell_model_experiments.params.params import *


def generate_dir(expected_path, subfolder="", args=None):

    if len(subfolder) == 0:
        # See if folder is present
        dir_exists = os.path.isdir(expected_path)

        if not dir_exists:
            os.makedirs(expected_path)

        subfolder = expected_path
    else:
        # Check if path exists
        expected_path = str(Path(expected_path, subfolder))
        dir_exists = os.path.isdir(expected_path)

        if not dir_exists:
            os.makedirs(expected_path)

    return expected_path


def generate_header(args, n_data=0, append_extra=""):
    arg_str_list = [f"{key}={value}" for key, value in args.items()]
    arg_str = ", ".join(arg_str_list)
    header = (
        arg_str
        + f", n_f={n_forcing}, dt={dt}, epsilon={epsilon}, "
        + f"lambda={lambda_const}, N_data={n_data}, "
        + f"sample_rate={sample_rate}, "
        + f"experiment={LICENCE}"
        + append_extra
    )

    return header


def convert_arguments_to_string(args):
    temp_time_to_run = "{:.2e}".format(args["time_to_run"])
    temp_forcing = "{:.1f}".format(args["forcing"])
    temp_ny = "{:.2e}".format(args["ny"])

    arguments = {
        "time_to_run": temp_time_to_run,
        "forcing": temp_forcing,
        "ny": temp_ny,
    }

    return arguments


def save_data(data_out, subfolder="", prefix="", perturb_position=None, args=None):
    """Save the data to disc."""

    if args is None:
        print("Please supply an argument dictionary to save data.")
        exit()

    # Prepare variables to be used when saving
    n_data = data_out.shape[0]
    temp_args = convert_arguments_to_string(args)

    # expected_path = generate_dir(expected_name, subfolder=subfolder, args=args)

    if args["ref_run"]:
        subsubfolder = "ref_data"
        # Generate path if not existing
        expected_path = (
            f"data/ny{temp_args['ny']}_t{temp_args['time_to_run']}"
            + f"_n_f{n_forcing}_f{temp_args['forcing']}"
        )
        expected_path = generate_dir(expected_path + f"/{subsubfolder}", args=args)

        prefix = "ref_"

        ref_filename_extra = f"_rec{args['record_id']}"

        ref_data_info_name = (
            f"{expected_path}/ref_data_info_ny"
            + f"{temp_args['ny']}_t{temp_args['time_to_run']}"
            + f"_n_f{n_forcing}_f{temp_args['forcing']}.txt"
        )
        arg_str_list = [f"{key}={value}" for key, value in args.items()]
        arg_str = ", ".join(arg_str_list)
        info_line = (
            arg_str
            + f", n_f={n_forcing}, dt={dt}, epsilon={epsilon}, "
            + f"lambda={lambda_const}, "
            + f"sample_rate={sample_rate}"
        )
        with open(ref_data_info_name, "w") as file:
            file.write(info_line)

        ref_header_extra = f", rec_id={args['record_id']}"
        header = generate_header(args, n_data=n_data, append_extra=ref_header_extra)

    else:
        ref_filename_extra = ""
        subsubfolder = args["perturb_folder"]

        # Generate path if not existing
        expected_path = generate_dir(Path(args["path"], subsubfolder), args=args)

        if perturb_position is not None:
            perturb_header_extra = f", perturb_pos={int(perturb_position)}"
            header = generate_header(
                args, n_data=n_data, append_extra=perturb_header_extra
            )

    # Save data
    np.savetxt(
        f"{expected_path}/{prefix}udata_ny{temp_args['ny']}_t{temp_args['time_to_run']}"
        + f"_n_f{n_forcing}_f{temp_args['forcing']}{ref_filename_extra}.csv",
        data_out,
        delimiter=",",
        header=header,
    )


def save_perturb_info(args=None):
    """Save info textfile about the perturbation runs"""

    temp_args = convert_arguments_to_string(args)

    expected_path = generate_dir(Path(args["path"], args["perturb_folder"]), args=args)

    # Prepare filename
    perturb_data_info_name = Path(
        expected_path,
        f"perturb_data_info_ny{temp_args['ny']}_t{temp_args['time_to_run']}"
        + f"_n_f{n_forcing}_f{temp_args['forcing']}.txt",
    )

    # Check if path already exists
    dir_exists = os.path.isfile(perturb_data_info_name)
    if dir_exists:
        print("Perturb info not saved, since file already exists")
        return

    print("Saving perturb data info textfile\n")

    # Prepare line to write
    info_line = (
        f"f={args['forcing']}, n_f={n_forcing}, n_ny={args['ny_n']}, "
        + f"ny={args['ny']}, time={args['time_to_run']}, dt={dt}, epsilon={epsilon}, "
        + f"lambda={lambda_const}, "
        + f"burn_in_time={args['burn_in_time']}, "
        + f"sample_rate={sample_rate}, eigen_perturb={args['eigen_perturb']}, "
        + f"seed_mode={args['seed_mode']}, "
        + f"single_shell_perturb={args['single_shell_perturb']}, "
        + f"start_time_offset={args['start_time_offset']}"
    )

    # Write to file
    with open(str(perturb_data_info_name), "w") as file:
        file.write(info_line)


def save_lorentz_block_data(
    perturb_data, subfolder="", prefix="", perturb_position=None, args=None
):
    # Prepare data arrays
    num_forecasts = int(args["time_to_run"] / args["start_time_offset"])

    # Fill in data
    # NOTE: [start + 1: end + 2: step]:
    # start + 1: to offset index to save forecast for the given day and not last
    # datapoint before
    # end + 2: +1 for same reason as for start + 1. +1 to have endpoint true of slice
    slice = np.s_[
        int(args["start_time_offset"] * sample_rate / dt)
        + 1 : int(args["start_time_offset"] * sample_rate / dt * num_forecasts)
        + 2 : int(args["start_time_offset"] * sample_rate / dt)
    ]
    data_out = perturb_data[slice, :]

    expected_path = generate_dir(
        args["path"], subfolder=f"{args['perturb_folder']}", args=args
    )
    if perturb_position is not None:
        lorentz_header_extra = f", perturb_pos={int(perturb_position)}"
        header = generate_header(
            args, n_data=num_forecasts, append_extra=lorentz_header_extra
        )

    temp_args = convert_arguments_to_string(args)

    # Save data
    np.savetxt(
        f"{expected_path}/{prefix}udata_ny{temp_args['ny']}_t{temp_args['time_to_run']}"
        + f"_n_f{n_forcing}_f{temp_args['forcing']}.csv",
        data_out,
        delimiter=",",
        header=header,
    )
