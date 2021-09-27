import sys

sys.path.append("..")
import numpy as np
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
import general.utils.saving.save_data_funcs as g_save
from general.params.model_licences import Models
from config import MODEL

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params


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
        int(args["start_time_offset"] * params.tts)
        + 1 : int(args["start_time_offset"] * params.tts * num_forecasts)
        + 2 : int(args["start_time_offset"] * params.tts)
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

    if MODEL == Models.SHELL_MODEL:
        out_name = (
            f"udata_ny{temp_args['ny']}_t{temp_args['time_to_run']}"
            + f"_n_f{params.n_forcing}_f{temp_args['forcing']}.csv"
        )

    elif MODEL == Models.LORENTZ63:
        out_name = (
            f"udata_sig{temp_args['sigma']}"
            + f"_t{temp_args['time_to_run']}"
            + f"_b{temp_args['b_const']}_r{temp_args['r_const']}.csv"
        )

    # Save data
    np.savetxt(
        f"{expected_path}/{prefix}{out_name}",
        data_out,
        delimiter=",",
        header=header,
    )
