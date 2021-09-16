from numba import njit, types
from shell_model_experiments.params.params import *


@njit(
    (
        types.Array(types.float64, 1, "C", readonly=True),
        types.Array(types.float64, 1, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
    ),
    cache=True,
)
def derivative_evaluator(x_old=None, dx=None, derivMatrix=None):
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
    # Update the derivMatrix
    derivMatrix[1, 2] = -x_old[0]
    derivMatrix[2, 0] = x_old[1]

    # Calculate change in u (du)
    dx = derivMatrix @ x_old

    return dx


@njit(
    types.Array(types.float64, 1, "C", readonly=False)(
        types.Array(types.float64, 1, "C", readonly=False),
        types.float64,
        types.Array(types.float64, 1, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
    ),
    cache=True,
)
def runge_kutta4_vec(y0=0, h=1, dx=None, derivMatrix=None):
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
    k1 = h * derivative_evaluator(x_old=y0, dx=dx, derivMatrix=derivMatrix)
    k2 = h * derivative_evaluator(x_old=y0 + 1 / 2 * k1, dx=dx, derivMatrix=derivMatrix)
    k3 = h * derivative_evaluator(x_old=y0 + 1 / 2 * k2, dx=dx, derivMatrix=derivMatrix)
    k4 = h * derivative_evaluator(x_old=y0 + k3, dx=dx, derivMatrix=derivMatrix)

    # Update y
    y0 = y0 + 1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4

    return y0
