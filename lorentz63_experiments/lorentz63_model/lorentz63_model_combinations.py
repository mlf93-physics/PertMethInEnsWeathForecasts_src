import numpy as np
from general.utils.module_import.type_import import *
import lorentz63_experiments.params.params as l63_params
from lorentz63_experiments.lorentz63_model.tl_lorentz63 import (
    run_model as l63_tl_model,
)
from lorentz63_experiments.lorentz63_model.atl_lorentz63 import (
    run_model as l63_atl_model,
)


def l63_tl_atl_model(
    u_perturb: np.ndarray,
    u_ref: np.ndarray,
    jacobian_matrix: np.ndarray,
    lorentz_matrix: np.ndarray,
    data_out: np.ndarray,
    args: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the TL model followed by the ATL model, i.e. running forth and back

    Parameters
    ----------
    u_perturb : np.ndarray
        The perturbation velocity vector to start on
    u_ref : np.ndarray
        The start reference velocity vector
    jacobian_matrix : np.ndarray
        The jacobian matrix
    lorentz_matrix : np.ndarray
        The Lorentz matrix
    data_out : np.ndarray
        The array to store the model solutions
    args : dict
        Run-time arguments

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
    l63_tl_model(
        u_perturb,
        u_ref,
        lorentz_matrix,
        jacobian_matrix,
        data_out,
        args["Nt"] + args["endpoint"] * 1,
        r_const=args["r_const"],
        raw_perturbation=True,
    )

    u_tl_out = data_out[-1, 1:].T

    # Run ATL model one time step
    l63_atl_model(
        np.reshape(data_out[-1, 1:].T, (l63_params.sdim, 1)),
        u_ref,
        lorentz_matrix,
        jacobian_matrix,
        data_out,
        args["Nt"] + args["endpoint"] * 1,
        r_const=args["r_const"],
        raw_perturbation=True,
    )

    u_atl_out = data_out[0, 1:].T

    return u_tl_out, u_atl_out, u_init_perturb
