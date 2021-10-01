import sys

sys.path.append("..")
from numba import njit, types
from config import NUMBA_CACHE


@njit(
    (
        types.Array(types.float64, 1, "C", readonly=True),
        types.Array(types.float64, 1, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
    ),
    cache=NUMBA_CACHE,
)
def derivative_evaluator(x_old=None, dx=None, deriv_matrix=None):
    """Derivative evaluator used in the Runge-Kutta method.

    Calculates the derivatives in the lorentz model.

    Parameters
    ----------
    x_old : ndarray
        The previous lorentz position array
    dx : ndarray
        A helper array used to store the current derivative of the lorentz
        positions. Updated at each call to this function.

    Returns
    -------
    dx : ndarray
        The updated derivative of the lorentz positions

    """
    # Update the deriv_matrix
    deriv_matrix[1, 2] = -x_old[0]
    deriv_matrix[2, 0] = x_old[1]

    # Calculate change in u (du)
    dx = deriv_matrix @ x_old

    return dx


@njit(
    types.Array(types.float64, 1, "C", readonly=False)(
        types.Array(types.float64, 1, "C", readonly=False),
        types.float64,
        types.Array(types.float64, 1, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
    ),
    cache=NUMBA_CACHE,
)
def runge_kutta4_vec(y0=0, h=1, dx=None, deriv_matrix=None):
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
    k1 = h * derivative_evaluator(x_old=y0, dx=dx, deriv_matrix=deriv_matrix)
    k2 = h * derivative_evaluator(
        x_old=y0 + 1 / 2 * k1, dx=dx, deriv_matrix=deriv_matrix
    )
    k3 = h * derivative_evaluator(
        x_old=y0 + 1 / 2 * k2, dx=dx, deriv_matrix=deriv_matrix
    )
    k4 = h * derivative_evaluator(x_old=y0 + k3, dx=dx, deriv_matrix=deriv_matrix)

    # Update y
    y0 = y0 + 1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4

    return y0
