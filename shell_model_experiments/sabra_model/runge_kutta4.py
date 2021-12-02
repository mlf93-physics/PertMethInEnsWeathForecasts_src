import sys

sys.path.append("..")
from numba import njit, types, typeof
from shell_model_experiments.params.params import *
from shell_model_experiments.params.params import PAR, ParamsStructType
import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
import general.utils.custom_decorators as c_dec

import config as cfg


@njit(
    (types.Array(types.complex128, 1, "C", readonly=False))(
        types.Array(types.complex128, 1, "C", readonly=True),
        types.float64,
        typeof(PAR),
    ),
    cache=cfg.NUMBA_CACHE,
)
def derivative_evaluator(
    u_old: np.ndarray = None,
    forcing: float = None,
    PAR: ParamsStructType = None,
):
    """Derivative evaluator used in the Runge-Kutta method.

    Calculates the non-linear part of the derivative of the shell velocities.

    Parameters
    ----------
    u_old : ndarray
        The previous shell velocity array

    Returns
    -------
    du : ndarray
        The updated derivative of the shell velocities

    """
    # Calculate change in u (du)
    PAR.du_array[PAR.bd_size : -PAR.bd_size] = PAR.pre_factor * (
        u_old.conj()[PAR.bd_size + 1 : -PAR.bd_size + 1] * u_old[PAR.bd_size + 2 :]
        + PAR.factor2
        * u_old.conj()[PAR.bd_size - 1 : -PAR.bd_size - 1]
        * u_old[PAR.bd_size + 1 : -PAR.bd_size + 1]
        + PAR.factor3
        * u_old[: -PAR.bd_size - 2]
        * u_old[PAR.bd_size - 1 : -PAR.bd_size - 1]
    )

    # Apply forcing
    PAR.du_array[PAR.n_forcing + PAR.bd_size] += forcing
    return PAR.du_array


@njit(
    types.Array(types.complex128, 1, "C", readonly=False)(
        types.Array(types.complex128, 1, "C", readonly=False),
        types.float64,
        typeof(PAR),
    ),
    cache=cfg.NUMBA_CACHE,
)
def runge_kutta4(
    u_old: np.ndarray = 0,
    forcing: float = None,
    PAR: ParamsStructType = None,
) -> np.ndarray:
    """Performs the Runge-Kutta-4 integration of non-linear part of the shell velocities.

    Parameters
    ----------
    u_old : np.ndarray, optional
        The previous shell velocities, by default 0
    forcing : float, optional
        The forcing, by default None
    PAR : ParamsStructType, optional
        The parameter struct, by default None

    Returns
    -------
    np.ndarray
        The integrated shell velocities
    """
    # Calculate the k's
    k1 = PAR.dt * derivative_evaluator(u_old=u_old, forcing=forcing, PAR=PAR)
    k2 = PAR.dt * derivative_evaluator(
        u_old=u_old + 1 / 2 * k1, forcing=forcing, PAR=PAR
    )
    k3 = PAR.dt * derivative_evaluator(
        u_old=u_old + 1 / 2 * k2, forcing=forcing, PAR=PAR
    )
    k4 = PAR.dt * derivative_evaluator(u_old=u_old + k3, forcing=forcing, PAR=PAR)

    # Update y
    u_old = u_old + 1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4

    return u_old


@njit(
    types.Array(types.complex128, 1, "C", readonly=False)(
        types.Array(types.complex128, 1, "C", readonly=True),
        types.Array(types.complex128, 1, "C", readonly=True),
        types.float64,
        types.float64,
        typeof(PAR),
        types.Array(types.complex128, 2, "C", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
    ),
    cache=cfg.NUMBA_CACHE,
)
def tl_derivative_evaluator(
    u_old: np.ndarray = None,
    u_ref: np.ndarray = None,
    diff_exponent: float = None,
    local_ny: float = None,
    PAR: ParamsStructType = None,
    J_matrix: np.ndarray = None,
    diagonal0: np.ndarray = None,
    diagonal1: np.ndarray = None,
    diagonal2: np.ndarray = None,
    diagonal_1: np.ndarray = None,
    diagonal_2: np.ndarray = None,
) -> np.ndarray:
    """Derivative evaluator used in the TL Runge-Kutta method.

    Calculates the derivatives in the lorentz model.

    Parameters
    ----------
    u_old : np.ndarray, optional
        The previous TL shell velocities, by default None
    u_ref : np.ndarray, optional
        The reference velocities, by default None
    diff_exponent : float, optional
        The diffusion exponent needed for hyper-diffusion, by default None
    local_ny : float, optional
        The local viscosity, by default None
    PAR : ParamsStructType, optional
        The parameter struct, by default None
    J_matrix : np.ndarray, optional
        The jacobian matrix array, by default None
    diagonal0 : np.ndarray, optional
        The diagonal of the jacobian at position k=0, by default None
    diagonal1 : np.ndarray, optional
        The diagonal of the jacobian at position k=1, by default None
    diagonal2 : np.ndarray, optional
        The diagonal of the jacobian at position k=2, by default None
    diagonal_1 : np.ndarray, optional
        The diagonal of the jacobian at position k=-1, by default None
    diagonal_2 : np.ndarray, optional
        The diagonal of the jacobian at position k=-2, by default None

    Returns
    -------
    np.ndarray
        The derivative of the TL shell model
    """

    # Update the jacobian matrix
    sh_nm_estimator.calc_jacobian(
        u_ref,
        diff_exponent,
        local_ny,
        PAR,
        diagonal0,
        diagonal1,
        diagonal2,
        diagonal_1,
        diagonal_2,
    )

    # Calculate change in u (du_array)
    PAR.du_array[PAR.bd_size : -PAR.bd_size] = (
        J_matrix @ u_old[PAR.bd_size : -PAR.bd_size]
    )

    return PAR.du_array


@njit(
    types.UniTuple(types.Array(types.complex128, 1, "C", readonly=False), 2,)(
        types.Array(types.complex128, 1, "C", readonly=False),
        types.Array(types.complex128, 1, "C", readonly=True),
        types.float64,
        types.float64,
        types.float64,
        typeof(PAR),
        types.Array(types.complex128, 2, "C", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
    ),
    cache=cfg.NUMBA_CACHE,
)
def tl_runge_kutta4(
    u_tl_old: np.ndarray = 0,
    u_ref_old: np.ndarray = None,
    diff_exponent: float = None,
    local_ny: float = None,
    forcing: float = None,
    PAR: ParamsStructType = None,
    J_matrix: np.ndarray = None,
    diagonal0: np.ndarray = None,
    diagonal1: np.ndarray = None,
    diagonal2: np.ndarray = None,
    diagonal_1: np.ndarray = None,
    diagonal_2: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs the Runge-Kutta-4 integration of the tangent linear Sabra shell
    model.

    Parameters
    ----------
    u_tl_old : np.ndarray
        The previous perturbation velocities, by default None
    u_ref_old : np.ndarray
        The previous reference/basis flow velocity vector, by default None
    diff_exponent : float, optional
        The exponent of the diffusion term, by default None
    local_ny : float, optional
        The local viscosity, by default None
    forcing : float, optional
        The forcing of the flow, by default None
    PAR : ParamsStructType, optional
        Parameters for the shell model, by default None
    J_matrix : np.ndarray, optional
        The jacobian matrix array, by default None
    diagonal0 : np.ndarray, optional
        The diagonal of the jacobian at position k=0, by default None
    diagonal1 : np.ndarray, optional
        The diagonal of the jacobian at position k=1, by default None
    diagonal2 : np.ndarray, optional
        The diagonal of the jacobian at position k=2, by default None
    diagonal_1 : np.ndarray, optional
        The diagonal of the jacobian at position k=-1, by default None
    diagonal_2 : np.ndarray, optional
        The diagonal of the jacobian at position k=-2, by default None

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (
            u_tl_old : The integrated perturbation velocities
            u_ref_old : The integrated reference/basis velocities
        )
    """

    # Calculate the k's of the non-linear model
    Y1 = u_ref_old
    k1 = PAR.dt * derivative_evaluator(u_old=Y1, forcing=forcing, PAR=PAR)
    Y2 = u_ref_old + 1 / 2 * k1
    k2 = PAR.dt * derivative_evaluator(u_old=Y2, forcing=forcing, PAR=PAR)
    Y3 = u_ref_old + 1 / 2 * k2
    k3 = PAR.dt * derivative_evaluator(u_old=Y3, forcing=forcing, PAR=PAR)
    Y4 = u_ref_old + k3
    k4 = PAR.dt * derivative_evaluator(u_old=Y4, forcing=forcing, PAR=PAR)

    # Calculate the k's
    l1 = PAR.dt * tl_derivative_evaluator(
        u_old=u_tl_old,
        u_ref=Y1,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        PAR=PAR,
        J_matrix=J_matrix,
        diagonal0=diagonal0,
        diagonal1=diagonal1,
        diagonal2=diagonal2,
        diagonal_1=diagonal_1,
        diagonal_2=diagonal_2,
    )
    l2 = PAR.dt * tl_derivative_evaluator(
        u_old=u_tl_old + 1 / 2 * l1,
        u_ref=Y2,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        PAR=PAR,
        J_matrix=J_matrix,
        diagonal0=diagonal0,
        diagonal1=diagonal1,
        diagonal2=diagonal2,
        diagonal_1=diagonal_1,
        diagonal_2=diagonal_2,
    )
    l3 = PAR.dt * tl_derivative_evaluator(
        u_old=u_tl_old + 1 / 2 * l2,
        u_ref=Y3,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        PAR=PAR,
        J_matrix=J_matrix,
        diagonal0=diagonal0,
        diagonal1=diagonal1,
        diagonal2=diagonal2,
        diagonal_1=diagonal_1,
        diagonal_2=diagonal_2,
    )
    l4 = PAR.dt * tl_derivative_evaluator(
        u_old=u_tl_old + l3,
        u_ref=Y4,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        PAR=PAR,
        J_matrix=J_matrix,
        diagonal0=diagonal0,
        diagonal1=diagonal1,
        diagonal2=diagonal2,
        diagonal_1=diagonal_1,
        diagonal_2=diagonal_2,
    )

    # Update u_tl_old
    u_tl_old = u_tl_old + 1 / 6 * (l1 + 2 * (l2 + l3) + l4)
    # Update u_ref_old
    u_ref_old = u_ref_old + 1 / 6 * (k1 + 2 * (k2 + k3) + k4)

    return u_tl_old, u_ref_old


@njit(
    types.Array(types.complex128, 1, "C", readonly=False)(
        types.Array(types.complex128, 1, "C", readonly=True),
        types.Array(types.complex128, 1, "C", readonly=True),
        types.float64,
        types.float64,
        typeof(PAR),
        types.Array(types.complex128, 2, "C", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
    ),
    cache=cfg.NUMBA_CACHE,
)
def atl_derivative_evaluator(
    u_atl_old: np.ndarray = None,
    u_ref: np.ndarray = None,
    diff_exponent: float = None,
    local_ny: float = None,
    PAR: ParamsStructType = None,
    J_matrix: np.ndarray = None,
    diagonal0: np.ndarray = None,
    diagonal1: np.ndarray = None,
    diagonal2: np.ndarray = None,
    diagonal_1: np.ndarray = None,
    diagonal_2: np.ndarray = None,
) -> np.ndarray:
    """Derivative evaluator used in the ATL Runge-Kutta method.

    Calculates the derivatives in the ATL shell model.

    Parameters
    ----------
    u_atl_old : np.ndarray, optional
        The previous adjoint of the TL shell velocities, by default None
    u_ref : np.ndarray, optional
        The reference velocities, by default None
    diff_exponent : float, optional
        The exponent of the diffusion term, by default None
    local_ny : float, optional
        The local viscosity, by default None
    forcing : float, optional
        The forcing of the flow, by default None
    PAR : ParamsStructType, optional
        Parameters for the shell model, by default None
    J_matrix : np.ndarray, optional
        The jacobian matrix array, by default None
    diagonal0 : np.ndarray, optional
        The diagonal of the jacobian at position k=0, by default None
    diagonal1 : np.ndarray, optional
        The diagonal of the jacobian at position k=1, by default None
    diagonal2 : np.ndarray, optional
        The diagonal of the jacobian at position k=2, by default None
    diagonal_1 : np.ndarray, optional
        The diagonal of the jacobian at position k=-1, by default None
    diagonal_2 : np.ndarray, optional
        The diagonal of the jacobian at position k=-2, by default None

    Returns
    -------
    ndarray
        The updated adjoint derivative of the shell velocities
    """
    # Update the adjoint jacobian matrix
    adjoint_jac_matrix = sh_nm_estimator.calc_adjoint_jacobian(
        u_ref,
        diff_exponent,
        local_ny,
        PAR,
        J_matrix,
        diagonal0,
        diagonal1,
        diagonal2,
        diagonal_1,
        diagonal_2,
    )

    # Calculate change in u (du_array)
    PAR.du_array[PAR.bd_size : -PAR.bd_size] = (
        adjoint_jac_matrix @ u_atl_old[PAR.bd_size : -PAR.bd_size]
    )

    return PAR.du_array


@njit(
    types.Array(types.complex128, 1, "C", readonly=False)(
        types.Array(types.complex128, 1, "C", readonly=False),
        types.Array(types.complex128, 1, "C", readonly=True),
        types.float64,
        types.float64,
        types.float64,
        typeof(PAR),
        types.Array(types.complex128, 2, "C", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
        types.Array(types.complex128, 1, "A", readonly=False),
    ),
    cache=cfg.NUMBA_CACHE,
)
def atl_runge_kutta4(
    u_atl_old: np.ndarray = 0,
    u_ref_old: np.ndarray = None,
    diff_exponent: float = None,
    local_ny: float = None,
    forcing: float = None,
    PAR: ParamsStructType = None,
    J_matrix: np.ndarray = None,
    diagonal0: np.ndarray = None,
    diagonal1: np.ndarray = None,
    diagonal2: np.ndarray = None,
    diagonal_1: np.ndarray = None,
    diagonal_2: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs the Runge-Kutta-4 integration of the adjoint tangent linear Sabra shell
    model.

    Parameters
    ----------
    u_atl_old : np.ndarray
        The previous perturbation velocities, by default None
    u_ref_old : np.ndarray
        The previous reference/basis flow velocity vector, by default None
    diff_exponent : float, optional
        The exponent of the diffusion term, by default None
    local_ny : float, optional
        The local viscosity, by default None
    forcing : float, optional
        The forcing of the flow, by default None
    PAR : ParamsStructType, optional
        Parameters for the shell model, by default None
    J_matrix : np.ndarray, optional
        The jacobian matrix array, by default None
    diagonal0 : np.ndarray, optional
        The diagonal of the jacobian at position k=0, by default None
    diagonal1 : np.ndarray, optional
        The diagonal of the jacobian at position k=1, by default None
    diagonal2 : np.ndarray, optional
        The diagonal of the jacobian at position k=2, by default None
    diagonal_1 : np.ndarray, optional
        The diagonal of the jacobian at position k=-1, by default None
    diagonal_2 : np.ndarray, optional
        The diagonal of the jacobian at position k=-2, by default None

    Returns
    -------
    np.ndarray
        u_atl_old : The integrated adjoint perturbation velocities
    """

    # Calculate the k's of the non-linear model
    Y1 = u_ref_old
    k1 = PAR.dt * derivative_evaluator(u_old=Y1, forcing=forcing, PAR=PAR)
    Y2 = u_ref_old + 1 / 2 * k1
    k2 = PAR.dt * derivative_evaluator(u_old=Y2, forcing=forcing, PAR=PAR)
    Y3 = u_ref_old + 1 / 2 * k2
    k3 = PAR.dt * derivative_evaluator(u_old=Y3, forcing=forcing, PAR=PAR)
    Y4 = u_ref_old + k3

    # Calculate the l's of the adjoint tangent linear model
    l4 = atl_derivative_evaluator(
        u_atl_old=u_atl_old,
        u_ref=Y4,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        PAR=PAR,
        J_matrix=J_matrix,
        diagonal0=diagonal0,
        diagonal1=diagonal1,
        diagonal2=diagonal2,
        diagonal_1=diagonal_1,
        diagonal_2=diagonal_2,
    )

    l3 = atl_derivative_evaluator(
        u_atl_old=u_atl_old + 1 / 2 * PAR.dt * l4,
        u_ref=Y3,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        PAR=PAR,
        J_matrix=J_matrix,
        diagonal0=diagonal0,
        diagonal1=diagonal1,
        diagonal2=diagonal2,
        diagonal_1=diagonal_1,
        diagonal_2=diagonal_2,
    )

    l2 = atl_derivative_evaluator(
        u_atl_old=u_atl_old + 1 / 2 * PAR.dt * l3,
        u_ref=Y2,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        PAR=PAR,
        J_matrix=J_matrix,
        diagonal0=diagonal0,
        diagonal1=diagonal1,
        diagonal2=diagonal2,
        diagonal_1=diagonal_1,
        diagonal_2=diagonal_2,
    )

    l1 = atl_derivative_evaluator(
        u_atl_old=u_atl_old + PAR.dt * l2,
        u_ref=Y1,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        PAR=PAR,
        J_matrix=J_matrix,
        diagonal0=diagonal0,
        diagonal1=diagonal1,
        diagonal2=diagonal2,
        diagonal_1=diagonal_1,
        diagonal_2=diagonal_2,
    )

    # Update u_atl_old
    u_atl_old = u_atl_old + 1 / 6 * PAR.dt * (l1 + 2 * (l2 + l3) + l4)

    return u_atl_old
