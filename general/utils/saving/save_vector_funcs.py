import pathlib as pl

import config as cfg
import general.utils.saving.save_utils as g_save_utils
import lorentz63_experiments.params.params as l63_params
import numpy as np
from shell_model_experiments.params.params import ParamsStructType
from shell_model_experiments.params.params import PAR as PAR_SH
import shell_model_experiments.utils.special_params as sh_sparams
import lorentz63_experiments.params.special_params as l63_sparams
from general.params.experiment_licences import Experiments as EXP
from general.params.model_licences import Models

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    params = PAR_SH
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    params = l63_params
    sparams = l63_sparams


def save_vector_unit(
    vectors: np.ndarray,
    perturb_position: int = None,
    unit: int = 0,
    args: dict = None,
    exp_setup: dict = None,
    characteristic_values: np.ndarray = None,
) -> None:
    """Save a vector unit to disk (e.g. BV or Lyapunov unit)

    Parameters
    ----------
    vectors : np.ndarray((n_vectors, sdim))
        The vector data to be saved
    characteristic_values : np.ndarray((n_vectors))
        The characteristic values of the vectors, i.e. singular values for singular
        vectors, or variances for BV-EOF vectors.
    perturb_position : int, optional
        The index position of the vector, by default None
    unit : int, optional
        The unit number, by default 0
    args : dict, optional
        Run-time arguments, by default None
    exp_setup : dict, optional
        Experiment setup, by default None
    """
    # Prepare variables to be used when saving
    n_data = vectors.shape[0]

    # Generate path if not existing
    expected_path = g_save_utils.generate_dir(
        pl.Path(args["datapath"], args["out_exp_folder"]), args=args
    )
    # Calculate position of when the vector is to be valid
    if cfg.LICENCE == EXP.BREEDING_VECTORS:
        val_pos = int(
            perturb_position
            + exp_setup["n_cycles"] * exp_setup["integration_time"] * params.tts
        )
    elif cfg.LICENCE == EXP.LYAPUNOV_VECTORS:
        val_pos = int(perturb_position + exp_setup["integration_time"] * params.tts)

    elif cfg.LICENCE == EXP.SINGULAR_VECTORS or EXP.BREEDING_EOF_VECTORS:
        val_pos = int(perturb_position)

    if perturb_position is not None:
        perturb_header_extra = (
            f"perturb_pos={int(perturb_position)}, unit={unit}"
            + f", val_pos={val_pos}, "
        )
        header = g_save_utils.generate_header(
            args,
            n_data=n_data,
            append_extra=perturb_header_extra,
            append_options=["licence"],
        )

    # Generate out file name
    stand_data_name = g_save_utils.generate_standard_data_name(args)
    out_name = f"_{stand_data_name}"

    prefix = cfg.LICENCE.name.lower()

    suffix = f"_unit{unit}"

    if characteristic_values is not None:
        concatenated_data_out = np.concatenate(
            [characteristic_values[:, np.newaxis], vectors], axis=1
        )
    else:
        concatenated_data_out = vectors

    # Save vectors
    np.savetxt(
        pl.Path(expected_path, f"{prefix}{out_name}{suffix}.csv"),
        concatenated_data_out,
        delimiter=",",
        header=header,
    )
