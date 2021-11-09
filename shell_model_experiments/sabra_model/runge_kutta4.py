import sys

sys.path.append("..")
from numba import njit, types
from shell_model_experiments.params.params import *
import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
import general.utils.custom_decorators as c_dec
import config as cfg


@njit(
    (
        types.Array(types.complex128, 1, "C", readonly=True),
        types.Array(types.complex128, 1, "C", readonly=False),
        types.float64,
        types.Array(types.complex128, 1, "C", readonly=True),
    ),
    cache=cfg.NUMBA_CACHE,
)
def derivative_evaluator(
    u_old: np.ndarray = None,
    du: np.ndarray = None,
    forcing: float = None,
    pre_factor: np.ndarray = None,
):
    """Derivative evaluator used in the Runge-Kutta method.

    Calculates the non-linear part of the derivative of the shell velocities.

    Parameters
    ----------
    u_old : ndarray
        The previous shell velocity array
    du : ndarray
        A helper array used to store the current derivative of the shell
        velocities. Updated at each call to this function.

    Returns
    -------
    du : ndarray
        The updated derivative of the shell velocities

    """
    # Calculate change in u (du)
    du[bd_size:-bd_size] = pre_factor * (
        u_old.conj()[bd_size + 1 : -bd_size + 1] * u_old[bd_size + 2 :]
        + factor2
        * u_old.conj()[bd_size - 1 : -bd_size - 1]
        * u_old[bd_size + 1 : -bd_size + 1]
        + factor3 * u_old[: -bd_size - 2] * u_old[bd_size - 1 : -bd_size - 1]
    )

    # Apply forcing
    du[n_forcing + bd_size] += forcing
    return du


@njit(
    types.Array(types.complex128, 1, "C", readonly=False)(
        types.Array(types.complex128, 1, "C", readonly=False),
        types.float64,
        types.Array(types.complex128, 1, "C", readonly=False),
        types.float64,
        types.Array(types.complex128, 1, "C", readonly=True),
    ),
    cache=cfg.NUMBA_CACHE,
)
def runge_kutta4(
    y0: np.ndarray = 0,
    h: float = 1,
    du: np.ndarray = None,
    forcing: float = None,
    pre_factor: np.ndarray = None,
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
    du : ndarray
        A helper array used to store the current derivative of the shell
        velocities.

    Returns
    -------
    y0 : ndarray
        The y array at x + h.

    """
    # Calculate the k's
    k1 = h * derivative_evaluator(
        u_old=y0, du=du, forcing=forcing, pre_factor=pre_factor
    )
    k2 = h * derivative_evaluator(
        u_old=y0 + 1 / 2 * k1, du=du, forcing=forcing, pre_factor=pre_factor
    )
    k3 = h * derivative_evaluator(
        u_old=y0 + 1 / 2 * k2, du=du, forcing=forcing, pre_factor=pre_factor
    )
    k4 = h * derivative_evaluator(
        u_old=y0 + k3, du=du, forcing=forcing, pre_factor=pre_factor
    )

    # Update y
    y0 = y0 + 1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4

    return y0


@njit(
    types.Array(types.complex128, 1, "C", readonly=False)(
        types.Array(types.complex128, 1, "C", readonly=True),
        types.Array(types.complex128, 1, "C", readonly=False),
        types.Array(types.complex128, 1, "C", readonly=True),
        types.float64,
        types.float64,
        types.Array(types.complex128, 2, "C", readonly=True),
        types.int32,
        types.Array(types.float64, 1, "C", readonly=True),
    ),
    cache=cfg.NUMBA_CACHE,
)
def tl_derivative_evaluator(
    u_old=None,
    du=None,
    u_ref=None,
    diff_exponent=None,
    local_ny=None,
    pre_factor=None,
    sdim_local=None,
    k_vec_temp_local=None,
):
    """Derivative evaluator used in the TL Runge-Kutta method.

    Calculates the derivatives in the lorentz model.

    Parameters
    ----------
    u_old : ndarray
        The previous lorentz velocity array
    du : ndarray
        A helper array used to store the current derivative of the lorentz
        velocities. Updated at each call to this function.

    Returns
    -------
    du : ndarray
        The updated derivative of the lorentz velocities

    """
    # Update the deriv_matrix
    j_matrix: np.ndarray = sh_nm_estimator.calc_jacobian(
        u_ref, diff_exponent, local_ny, pre_factor, sdim_local, k_vec_temp_local
    )

    # Calculate change in u (du)
    du[bd_size:-bd_size] = j_matrix @ u_old[bd_size:-bd_size]

    return du


@njit(
    types.Array(types.complex128, 1, "C", readonly=False)(
        types.Array(types.complex128, 1, "C", readonly=False),
        types.float64,
        types.Array(types.complex128, 1, "C", readonly=False),
        types.Array(types.complex128, 1, "C", readonly=True),
        types.float64,
        types.float64,
        types.Array(types.complex128, 2, "C", readonly=True),
        types.int32,
        types.Array(types.float64, 1, "C", readonly=True),
    ),
    cache=cfg.NUMBA_CACHE,
)
def tl_runge_kutta4(
    y0=0,
    h=1,
    du=None,
    u_ref=None,
    diff_exponent=None,
    local_ny=None,
    pre_factor=None,
    sdim_local=None,
    k_vec_temp_local=None,
):
    """Performs the Runge-Kutta-4 integration of the lorentz model.

    Parameters
    ----------
    x0 : ndarray
        The x array
    y0 : ndarray
        The y array
    h : float
        The x step to integrate over.
    du : ndarray
        A helper array used to store the current derivative of the shell
        velocities.

    Returns
    -------
    y0 : ndarray
        The y array at x + h.

    """
    # Calculate the k's
    k1 = h * tl_derivative_evaluator(
        u_old=y0,
        du=du,
        u_ref=u_ref,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        pre_factor=pre_factor,
        sdim_local=sdim_local,
        k_vec_temp_local=k_vec_temp_local,
    )
    k2 = h * tl_derivative_evaluator(
        u_old=y0 + 1 / 2 * k1,
        du=du,
        u_ref=u_ref,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        pre_factor=pre_factor,
        sdim_local=sdim_local,
        k_vec_temp_local=k_vec_temp_local,
    )
    k3 = h * tl_derivative_evaluator(
        u_old=y0 + 1 / 2 * k2,
        du=du,
        u_ref=u_ref,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        pre_factor=pre_factor,
        sdim_local=sdim_local,
        k_vec_temp_local=k_vec_temp_local,
    )
    k4 = h * tl_derivative_evaluator(
        u_old=y0 + k3,
        du=du,
        u_ref=u_ref,
        diff_exponent=diff_exponent,
        local_ny=local_ny,
        pre_factor=pre_factor,
        sdim_local=sdim_local,
        k_vec_temp_local=k_vec_temp_local,
    )

    # Update y
    y0 = y0 + 1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4

    return y0
