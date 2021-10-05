import pathlib as pl
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


def save_breed_vector_unit(
    breed_data, perturb_position=None, br_unit=0, args=None, exp_setup=None
):
    breed_data = breed_data.T

    # Prepare variables to be used when saving
    n_data = breed_data.shape[0]
    temp_args = g_save.convert_arguments_to_string(args)

    # Generate path if not existing
    expected_path = g_save.generate_dir(
        pl.Path(args["datapath"], args["exp_folder"]), args=args
    )
    # Calculate position of when the breed_vector is to be valid
    val_pos = int(
        perturb_position
        + exp_setup["n_cycles"] * exp_setup["time_per_cycle"] * params.tts
    )

    if perturb_position is not None:
        perturb_header_extra = (
            f", perturb_pos={int(perturb_position)}, br_unit={br_unit}"
            + f", val_pos={val_pos}"
        )
        header = g_save.generate_header(
            args, n_data=n_data, append_extra=perturb_header_extra
        )

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

    prefix = "breed_vectors"
    suffix = f"_br_unit{br_unit}"
    # Save data
    np.savetxt(
        pl.Path(expected_path, f"{prefix}{out_name}{suffix}.csv"),
        breed_data[:, params.u_slice],
        delimiter=",",
        header=header,
    )
