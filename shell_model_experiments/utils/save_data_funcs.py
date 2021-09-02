import os
import numpy as np
from pathlib import Path
from shell_model_experiments.params.params import *


def generate_dir(expected_name, subfolder="", args=None):

    if len(subfolder) == 0:
        expected_path = f"{expected_name}"

        # See if folder is present
        dir_exists = os.path.isdir(expected_path)

        if not dir_exists:
            os.mkdir(expected_path)

        subfolder = expected_name
    else:
        # Check if path exists
        expected_path = f"./data/{subfolder}"
        dir_exists = os.path.isdir(expected_path)

        if not dir_exists:
            os.mkdir(expected_path)

    return expected_path


def generate_header(args, n_data, append_extra=""):
    arg_str_list = [f"{key}={value}" for key, value in args.items()]
    arg_str = ", ".join(arg_str_list)
    header = (
        arg_str
        + f", n_f={n_forcing}, dt={dt}, epsilon={epsilon}, "
        + f"lambda={lambda_const}, N_data={n_data}, "
        + f"sample_rate={sample_rate}"
        + append_extra
    )

    return header


def save_data(data_out, subfolder="", prefix="", perturb_position=None, args=None):
    """Save the data to disc."""

    if args is None:
        print("Please supply an argument dictionary to save data.")
        exit()

    # Prepare variables to be used when saving
    n_data = data_out.shape[0]
    temp_time_to_run = "{:.2e}".format(args["time_to_run"])
    temp_forcing = "{:.1f}".format(args["forcing"])
    temp_ny = "{:.2e}".format(args["ny"])

    expected_name = (
        f"data/ny{temp_ny}_t{temp_time_to_run}" + f"_n_f{n_forcing}_f{temp_forcing}"
    )

    expected_path = generate_dir(expected_name, subfolder=subfolder, args=args)

    if args["ref_run"]:
        subsubfolder = "ref_data"
        # Generate path if not existing
        expected_path = generate_dir(expected_path + f"/{subsubfolder}", args=args)

        prefix = "ref_"

        ref_filename_extra = f"_rec{args['record_id']}"

        ref_data_info_name = (
            f"{expected_path}/ref_data_info_ny"
            + f"{temp_ny}_t{temp_time_to_run}"
            + f"_n_f{n_forcing}_f{temp_forcing}.txt"
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
        header = generate_header(args, n_data, append_extra=ref_header_extra)

    else:
        ref_filename_extra = ""
        subsubfolder = args["perturb_folder"]

        # Generate path if not existing
        expected_path = generate_dir(expected_path + f"/{subsubfolder}", args=args)

        if perturb_position is not None:
            perturb_header_extra = f", perturb_pos={int(perturb_position)}"
            header = generate_header(args, n_data, append_extra=perturb_header_extra)

    # Save data
    np.savetxt(
        f"{expected_path}/{prefix}udata_ny{temp_ny}_t{temp_time_to_run}"
        + f"_n_f{n_forcing}_f{temp_forcing}{ref_filename_extra}.csv",
        data_out,
        delimiter=",",
        header=header,
    )


def save_perturb_info(args=None):
    """Save info textfile about the perturbation runs"""

    temp_time_to_run = "{:.2e}".format(args["time_to_run"])
    temp_forcing = "{:.1f}".format(args["forcing"])
    temp_ny = "{:.2e}".format(args["ny"])

    # Prepare filename
    perturb_data_info_name = Path(
        args["path"],
        args["perturb_folder"],
        f"perturb_data_info_ny{temp_ny}_t{temp_time_to_run}"
        + f"_n_f{n_forcing}_f{temp_forcing}.txt",
    )

    # Check if path already exists
    dir_exists = os.path.isdir(perturb_data_info_name)
    if dir_exists:
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
    data_out, subfolder="", prefix="", perturb_position=None, args=None
):
    expected_path = generate_dir()
    n_data = 0
    header = generate_header(args, n_data, append_extra=perturb_header_extra)
