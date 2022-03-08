import numpy as np
from general.utils.module_import.type_import import *
from shell_model_experiments.sabra_model.tl_sabra_model import (
    run_model as sh_tl_model,
)
from shell_model_experiments.sabra_model.atl_sabra_model import (
    run_model as sh_atl_model,
)
from shell_model_experiments.params.params import PAR
from shell_model_experiments.params.params import ParamsStructType


def sh_tl_atl_model(
    u_perturb: np.ndarray,
    u_ref: np.ndarray,
    data_out: np.ndarray,
    args: dict,
    J_matrix: np.ndarray,
    diagonal0: np.ndarray,
    diagonal1: np.ndarray,
    diagonal2: np.ndarray,
    diagonal_1: np.ndarray,
    diagonal_2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the TL model followed by the ATL model, i.e. running forth and back

    Parameters
    ----------
    u_perturb : np.ndarray
        The perturbation velocity vector to start on
    u_ref : np.ndarray
        The start reference velocity vector
    data_out : np.ndarray
        The array to store the model solutions
    args : dict
        Run-time arguments
    J_matrix : np.ndarray
        The jacobian matrix array
    diagonal0 : np.ndarray
        The diagonal of the jacobian at position k=0
    diagonal1 : np.ndarray
        The diagonal of the jacobian at position k=1
    diagonal2 : np.ndarray
        The diagonal of the jacobian at position k=2
    diagonal_1 : np.ndarray
        The diagonal of the jacobian at position k=-1
    diagonal_2 : np.ndarray
        The diagonal of the jacobian at position k=-2

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (
            u_tl_out : The last velocity vector of the TL model solution
            u_atl_out : The last velocity vector of the ATL model solution
            u_init_perturb : The initial perturbation vector
        )
    """
    u_init_perturb = np.copy(u_perturb)

    # Run TL model one time step
    sh_tl_model(
        u_perturb,
        u_ref,
        data_out,
        args["Nt"] + args["endpoint"] * 1,
        args["ny"],
        args["diff_exponent"],
        args["forcing"],
        PAR,
        J_matrix,
        diagonal0,
        diagonal1,
        diagonal2,
        diagonal_1,
        diagonal_2,
        raw_perturbation=True,
    )

    u_tl_out = data_out[-1, 1:]

    # Run ATL model one time step
    sh_atl_model(
        np.pad(
            data_out[-1, 1:].T,
            pad_width=(PAR.bd_size, PAR.bd_size),
            mode="constant",
        ),
        u_ref,
        data_out,
        args["Nt"] + args["endpoint"] * 1,
        args["ny"],
        args["diff_exponent"],
        args["forcing"],
        PAR,
        J_matrix,
        diagonal0,
        diagonal1,
        diagonal2,
        diagonal_1,
        diagonal_2,
        raw_perturbation=True,
    )

    u_atl_out = data_out[0, 1:].T

    return u_tl_out, u_atl_out, u_init_perturb


def sh_atl_tl_model(
    u_perturb: np.ndarray,
    u_ref: np.ndarray,
    data_out: np.ndarray,
    args: dict,
    J_matrix: np.ndarray,
    diagonal0: np.ndarray,
    diagonal1: np.ndarray,
    diagonal2: np.ndarray,
    diagonal_1: np.ndarray,
    diagonal_2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the ATL model followed by the TL model, i.e. running back and forth

    Parameters
    ----------
    u_perturb : np.ndarray
        The perturbation velocity vector to start on
    u_ref : np.ndarray
        The start reference velocity vector
    data_out : np.ndarray
        The array to store the model solutions
    args : dict
        Run-time arguments
    J_matrix : np.ndarray
        The jacobian matrix array
    diagonal0 : np.ndarray
        The diagonal of the jacobian at position k=0
    diagonal1 : np.ndarray
        The diagonal of the jacobian at position k=1
    diagonal2 : np.ndarray
        The diagonal of the jacobian at position k=2
    diagonal_1 : np.ndarray
        The diagonal of the jacobian at position k=-1
    diagonal_2 : np.ndarray
        The diagonal of the jacobian at position k=-2

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (
            u_tl_out : The last velocity vector of the TL model solution
            u_atl_out : The last velocity vector of the ATL model solution
            u_init_perturb : The initial perturbation vector
        )
    """
    u_init_perturb = np.copy(u_perturb)

    # Run ATL model one time step
    sh_atl_model(
        u_perturb,
        u_ref,
        data_out,
        args["Nt"] + args["endpoint"] * 1,
        args["ny"],
        args["diff_exponent"],
        args["forcing"],
        PAR,
        J_matrix,
        diagonal0,
        diagonal1,
        diagonal2,
        diagonal_1,
        diagonal_2,
        raw_perturbation=True,
    )

    u_atl_out = data_out[0, 1:].T

    # Run TL model one time step
    sh_tl_model(
        np.pad(
            u_atl_out,
            pad_width=(PAR.bd_size, PAR.bd_size),
            mode="constant",
        ),
        u_ref,
        data_out,
        args["Nt"] + args["endpoint"] * 1,
        args["ny"],
        args["diff_exponent"],
        args["forcing"],
        PAR,
        J_matrix,
        diagonal0,
        diagonal1,
        diagonal2,
        diagonal_1,
        diagonal_2,
        raw_perturbation=True,
    )

    u_tl_out = data_out[-1, 1:]

    return u_tl_out, u_atl_out, u_init_perturb
