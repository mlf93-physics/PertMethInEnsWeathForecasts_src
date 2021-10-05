import sys

sys.path.append("..")
from numba import njit, types
import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
from config import NUMBA_CACHE


@njit(
    (
        types.Array(types.float64, 1, "C", readonly=True),
        types.Array(types.float64, 1, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
    ),
    cache=NUMBA_CACHE,
)
def derivative_evaluator(u_old=None, du=None, deriv_matrix=None):
    """Derivative evaluator used in the Runge-Kutta method.

    Calculates the derivatives in the lorentz model.

    Parameters
    ----------
    u_old : ndarray
        The previous lorentz velocitie array
    du : ndarray
        A helper array used to store the current derivative of the lorentz
        velocities. Updated at each call to this function.

    Returns
    -------
    du : ndarray
        The updated derivative of the lorentz velocities

    """
    # Update the deriv_matrix
    deriv_matrix[1, 2] = -u_old[0]
    deriv_matrix[2, 0] = u_old[1]

    # Calculate change in u (du)
    du = deriv_matrix @ u_old

    return du


@njit(
    types.Array(types.float64, 1, "C", readonly=False)(
        types.Array(types.float64, 1, "C", readonly=False),
        types.float64,
        types.Array(types.float64, 1, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
    ),
    cache=NUMBA_CACHE,
)
def runge_kutta4(y0=0, h=1, du=None, deriv_matrix=None):
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
    k1 = h * derivative_evaluator(u_old=y0, du=du, deriv_matrix=deriv_matrix)
    k2 = h * derivative_evaluator(
        u_old=y0 + 1 / 2 * k1, du=du, deriv_matrix=deriv_matrix
    )
    k3 = h * derivative_evaluator(
        u_old=y0 + 1 / 2 * k2, du=du, deriv_matrix=deriv_matrix
    )
    k4 = h * derivative_evaluator(u_old=y0 + k3, du=du, deriv_matrix=deriv_matrix)

    # Update y
    y0 = y0 + 1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4

    return y0


def tl_derivative_evaluator(
    u_old=None, du=None, u_ref=None, deriv_matrix=None, r_const=None
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
    deriv_matrix = l63_nm_estimator.calc_jacobian(deriv_matrix, u_ref, r_const=r_const)

    # Calculate change in u (du)
    du = deriv_matrix @ u_old

    return du


# @njit(
#     types.Array(types.float64, 1, "C", readonly=False)(
#         types.Array(types.float64, 1, "C", readonly=False),
#         types.float64,
#         types.Array(types.float64, 1, "C", readonly=False),
#         types.Array(types.float64, 2, "C", readonly=False),
#     ),
#     cache=NUMBA_CACHE,
# )
def tl_runge_kutta4(y0=0, h=1, du=None, u_ref=None, deriv_matrix=None, r_const=28):
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
        u_old=y0, du=du, u_ref=u_ref, deriv_matrix=deriv_matrix, r_const=r_const
    )
    k2 = h * tl_derivative_evaluator(
        u_old=y0 + 1 / 2 * k1,
        du=du,
        u_ref=u_ref,
        deriv_matrix=deriv_matrix,
        r_const=r_const,
    )
    k3 = h * tl_derivative_evaluator(
        u_old=y0 + 1 / 2 * k2,
        du=du,
        u_ref=u_ref,
        deriv_matrix=deriv_matrix,
        r_const=r_const,
    )
    k4 = h * tl_derivative_evaluator(
        u_old=y0 + k3, du=du, u_ref=u_ref, deriv_matrix=deriv_matrix, r_const=r_const
    )

    # Update y
    y0 = y0 + 1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4

    return y0
