import sys

sys.path.append("..")
from numba import njit, types, typeof
from shell_model_experiments.params.params import *
from shell_model_experiments.params.params import PAR, ParamsStructType
import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
import general.utils.custom_decorators as c_dec

# from shell_model_experiments.params.params import Params
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
    PAR=None,
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
    y0: np.ndarray = 0,
    forcing: float = None,
    PAR=None,
):
    """Performs the Runge-Kutta-4 integration of non-linear part of the shell velocities.

    Parameters
    ----------
    x0 : ndarray
        The x array
    y0 : ndarray
        The y array
    h : float
        The x step to integrate over.
    du_array : ndarray
        A helper array used to store the current derivative of the shell
        velocities.

    Returns
    -------
    y0 : ndarray
        The y array at x + h.

    """
    # Calculate the k's
    k1 = PAR.dt * derivative_evaluator(u_old=y0, forcing=forcing, PAR=PAR)
    k2 = PAR.dt * derivative_evaluator(u_old=y0 + 1 / 2 * k1, forcing=forcing, PAR=PAR)
    k3 = PAR.dt * derivative_evaluator(u_old=y0 + 1 / 2 * k2, forcing=forcing, PAR=PAR)
    k4 = PAR.dt * derivative_evaluator(u_old=y0 + k3, forcing=forcing, PAR=PAR)

    # Update y
    y0 = y0 + 1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4

    return y0


@njit(
    types.Array(types.complex128, 1, "C", readonly=False)(
        types.Array(types.complex128, 1, "C", readonly=True),
        types.Array(types.complex128, 1, "C", readonly=True),
        types.float64,
        types.float64,
        types.Array(types.complex128, 2, "C", readonly=True),
        typeof(PAR),
    ),
    cache=cfg.NUMBA_CACHE,
)
def tl_derivative_evaluator(
    u_old=None,
    u_ref=None,
    diff_exponent=None,
    local_ny=None,
    pre_factor_reshaped=None,
    PAR=None,
):
    """Derivative evaluator used in the TL Runge-Kutta method.

    Calculates the derivatives in the lorentz model.

    Parameters
    ----------
    u_old : ndarray
        The previous lorentz velocity array

    Returns
    -------
    du_array : ndarray
        The updated derivative of the lorentz velocities

    """
    # Update the jacobian matrix
    jacobian_matrix: np.ndarray = sh_nm_estimator.calc_jacobian(
        u_ref,
        diff_exponent,
        local_ny,
        pre_factor_reshaped,
        PAR,
    )

    # Calculate change in u (du_array)
    PAR.du_array[PAR.bd_size : -PAR.bd_size] = (
        jacobian_matrix @ u_old[PAR.bd_size : -PAR.bd_size]
    )

    return PAR.du_array


@njit(
    types.UniTuple(types.Array(types.complex128, 1, "C", readonly=False), 2,)(
        types.Array(types.complex128, 1, "C", readonly=False),
        types.Array(types.complex128, 1, "C", readonly=True),
        types.float64,
        types.float64,
        types.float64,
        types.Array(types.complex128, 2, "C", readonly=True),
        typeof(PAR),
    ),
    cache=cfg.NUMBA_CACHE,
)
def tl_runge_kutta4(
    u_tl_old: np.ndarray = 0,
    u_ref_old: np.ndarray = None,
    diff_exponent: float = None,
    local_ny: float = None,
    forcing: float = None,
    pre_factor_reshaped: np.ndarray = None,
    PAR: ParamsStructType = None,
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
    pre_factor_reshaped : np.ndarray, optional
        Helper array to carry out multiplication of the prefactor in the sabra
        shell model, by default None
    PAR : ParamsStructType, optional
        Parameters for the shell model, by default None

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
        pre_factor_reshaped=pre_factor_reshaped,
        PAR=PAR,
    )
    l2 = PAR.dt * tl_derivative_evaluator(
        u_old=u_tl_old + 1 / 2 * l1,
        u_ref=Y2,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        pre_factor_reshaped=pre_factor_reshaped,
        PAR=PAR,
    )
    l3 = PAR.dt * tl_derivative_evaluator(
        u_old=u_tl_old + 1 / 2 * l2,
        u_ref=Y3,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        pre_factor_reshaped=pre_factor_reshaped,
        PAR=PAR,
    )
    l4 = PAR.dt * tl_derivative_evaluator(
        u_old=u_tl_old + l3,
        u_ref=Y4,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        pre_factor_reshaped=pre_factor_reshaped,
        PAR=PAR,
    )

    # Update u_tl_old
    u_tl_old = u_tl_old + 1 / 6 * (l1 + 2 * (l2 + l3) + l4)
    # Update u_ref_old
    u_ref_old = u_ref_old + 1 / 6 * (k1 + 2 * (k2 + k3) + k4)

    return u_tl_old, u_ref_old
