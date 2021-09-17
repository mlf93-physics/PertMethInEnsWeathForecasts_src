import sys

sys.path.append("..")
import numpy as np
from shell_model_experiments.params.params import *
import general.utils.save_data_funcs as g_save


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
        int(args["start_time_offset"] * tts)
        + 1 : int(args["start_time_offset"] * tts * num_forecasts)
        + 2 : int(args["start_time_offset"] * tts)
    ]
    data_out = perturb_data[slice, :]

    expected_path = g_save.generate_dir(
        args["path"], subfolder=f"{args['perturb_folder']}", args=args
    )
    if perturb_position is not None:
        lorentz_header_extra = f", perturb_pos={int(perturb_position)}"
        header = g_save.generate_header(
            args, n_data=num_forecasts, append_extra=lorentz_header_extra
        )

    temp_args = g_save.convert_arguments_to_string(args)

    # Save data
    np.savetxt(
        f"{expected_path}/{prefix}udata_ny{temp_args['ny']}_t{temp_args['time_to_run']}"
        + f"_n_f{n_forcing}_f{temp_args['forcing']}.csv",
        data_out,
        delimiter=",",
        header=header,
    )
