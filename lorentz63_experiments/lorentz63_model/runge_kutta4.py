import sys

sys.path.append("..")
import numpy as np
from numba import njit, types
import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
from lorentz63_experiments.params.params import *
from general.utils.module_import.type_import import *
import config as cfg


@njit(
    types.Array(types.float64, 1, "C", readonly=False)(
        types.Array(types.float64, 1, "C", readonly=True),
        types.Array(types.float64, 2, "C", readonly=False),
    ),
    cache=cfg.NUMBA_CACHE,
)
def derivative_evaluator(
    u_old: np.ndarray = None,
    lorentz_matrix: np.ndarray = None,
):
    """Derivative evaluator used in the Runge-Kutta method.

    Calculates the derivatives in the lorentz model.

    Parameters
    ----------
    u_old : np.ndarray
        The previous lorentz velocitie array
    lorentz_matrix : np.ndarray
        The lorentz matrix, which enables direct calculation of dot(u) from
        matrix multiplication with u, by default None

    Returns
    -------
    du_array : np.ndarray
        The updated derivative of the lorentz velocities

    """
    # Update the lorentz_matrix
    lorentz_matrix[1, 2] = -u_old[0]
    lorentz_matrix[2, 0] = u_old[1]

    # Calculate change in u (du_array)
    du_array = lorentz_matrix @ u_old

    return du_array


@njit(
    types.Array(types.float64, 1, "C", readonly=False)(
        types.Array(types.float64, 1, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
        types.float64,
    ),
    cache=cfg.NUMBA_CACHE,
)
def runge_kutta4(
    y0: np.ndarray = 0,
    lorentz_matrix: np.ndarray = None,
    h: float = 1,
):
    """Performs the Runge-Kutta-4 integration of the lorentz model.

    Parameters
    ----------
    y0 : np.ndarray
        The y array
    h : float
        The x step to integrate over.
    lorentz_matrix : np.ndarray
        The lorentz matrix, which enables direct calculation of dot(u) from
        matrix multiplication with u, by default None

    Returns
    -------
    y0 : np.ndarray
        The y array at x + h.

    """
    # Calculate the k's
    k1 = h * derivative_evaluator(u_old=y0, lorentz_matrix=lorentz_matrix)
    k2 = h * derivative_evaluator(u_old=y0 + 1 / 2 * k1, lorentz_matrix=lorentz_matrix)
    k3 = h * derivative_evaluator(u_old=y0 + 1 / 2 * k2, lorentz_matrix=lorentz_matrix)
    k4 = h * derivative_evaluator(u_old=y0 + k3, lorentz_matrix=lorentz_matrix)

    # Update y
    y0 = y0 + 1 / 6 * k1 + 1 / 3 * k2 + 1 / 3 * k3 + 1 / 6 * k4

    return y0


@njit(
    (types.Array(types.float64, 1, "C", readonly=True))(
        types.Array(types.float64, 1, "C", readonly=True),
        types.Array(types.float64, 1, "C", readonly=True),
        types.Array(types.float64, 2, "C", readonly=False),
        types.float64,
    ),
    cache=cfg.NUMBA_CACHE,
)
def tl_derivative_evaluator(
    u_tl_old: np.ndarray = None,
    u_ref: np.ndarray = None,
    jacobian_matrix: np.ndarray = None,
    r_const: float = None,
) -> np.ndarray:
    """Derivative evaluator used in the TL Runge-Kutta method.

    Calculates the increment of the perturbation velocities in the TL lorentz model.

    Parameters
    ----------
    u_tl_old : ndarray
        The previous lorentz velocity array
    u_ref : np.ndarray
        The reference/basis state velocities, by default None
    jacobian_matrix : np.ndarray
        The jacobian of the Lorentz63 model, by default None
    r_const : float
        The constant r in the Lorentz63 model, by default 28

    Returns
    -------
    np.ndarray
        (
            du_array : The updated derivative of the lorentz velocities
        )
    """
    # Update the jacobian_matrix
    l63_nm_estimator.calc_jacobian(jacobian_matrix, u_ref, r_const=r_const)

    # Calculate change in u (du_array)
    du_array = jacobian_matrix @ u_tl_old

    return du_array


@njit(
    types.UniTuple(types.Array(types.float64, 1, "C", readonly=False), 2,)(
        types.Array(types.float64, 1, "C", readonly=False),
        types.Array(types.float64, 1, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
        types.float64,
        types.float64,
    ),
    cache=cfg.NUMBA_CACHE,
)
def tl_runge_kutta4(
    u_tl_old: np.ndarray = None,
    u_ref_old: np.ndarray = None,
    lorentz_matrix: np.ndarray = None,
    jacobian_matrix: np.ndarray = None,
    dt: float = 1,
    r_const: float = 28,
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs the Runge-Kutta-4 integration of the tangent linear Lorentz-63
    model.

    Parameters
    ----------
    u_tl_old : np.ndarray
        The previous perturbation velocities, by default None
    dt : float
        The time step, by default 1
    u_ref_old : np.ndarray
        The previous reference/basis flow velocity vector, by default None
    lorentz_matrix : np.ndarray
        The lorentz matrix, which enables direct calculation of dot(u) from
        matrix multiplication with u, by default None
    jacobian_matrix : np.ndarray
        The jacobian matrix of the Lorentz63 model, by default None
    r_const : float
        The constant r in the Lorentz63 model, by default 28

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
    k1 = dt * derivative_evaluator(u_old=Y1, lorentz_matrix=lorentz_matrix)
    Y2 = u_ref_old + 1 / 2 * k1
    k2 = dt * derivative_evaluator(u_old=Y2, lorentz_matrix=lorentz_matrix)
    Y3 = u_ref_old + 1 / 2 * k2
    k3 = dt * derivative_evaluator(u_old=Y3, lorentz_matrix=lorentz_matrix)
    Y4 = u_ref_old + k3
    k4 = dt * derivative_evaluator(u_old=Y4, lorentz_matrix=lorentz_matrix)

    # Calculate the k's of the tangent linear model
    l1 = tl_derivative_evaluator(
        u_tl_old=u_tl_old,
        u_ref=Y1,
        jacobian_matrix=jacobian_matrix,
        r_const=r_const,
    )
    l2 = tl_derivative_evaluator(
        u_tl_old=u_tl_old + 1 / 2 * dt * l1,
        u_ref=Y2,
        jacobian_matrix=jacobian_matrix,
        r_const=r_const,
    )
    l3 = tl_derivative_evaluator(
        u_tl_old=u_tl_old + 1 / 2 * dt * l2,
        u_ref=Y3,
        jacobian_matrix=jacobian_matrix,
        r_const=r_const,
    )
    l4 = tl_derivative_evaluator(
        u_tl_old=u_tl_old + dt * l3,
        u_ref=Y4,
        jacobian_matrix=jacobian_matrix,
        r_const=r_const,
    )

    # Update u_tl_old
    u_tl_old = u_tl_old + 1 / 6 * dt * (l1 + 2 * (l2 + l3) + l4)
    # Update u_ref_old
    u_ref_old = u_ref_old + 1 / 6 * (k1 + 2 * (k2 + k3) + k4)

    return u_tl_old, u_ref_old


@njit(
    (
        types.Array(types.float64, 2, "C", readonly=True),
        types.Array(types.float64, 1, "C", readonly=True),
        types.Array(types.float64, 2, "C", readonly=False),
        types.float64,
    ),
    cache=cfg.NUMBA_CACHE,
)
def atl_derivative_evaluator(
    u_atl_old: np.ndarray((sdim, 1)) = None,
    u_ref: np.ndarray = None,
    jacobian_matrix: np.ndarray = None,
    r_const: float = None,
) -> np.ndarray:
    """Derivative evaluator used in the TL Runge-Kutta method.

    Calculates the increment of the perturbation velocities in the TL lorentz model.

    Parameters
    ----------
    u_atl_old : ndarray
        The previous lorentz velocity array
    u_ref : np.ndarray
        The reference/basis state velocities, by default None
    jacobian_matrix : np.ndarray
        The jacobian of the Lorentz63 model, by default None
    r_const : float
        The constant r in the Lorentz63 model, by default 28

    Returns
    -------
    np.ndarray
        (
            du_array : The updated derivative of the lorentz velocities
        )
    """
    # Update the jacobian_matrix
    adjoint_jac_matrix = l63_nm_estimator.calc_adjoint_jacobian(
        jacobian_matrix, u_ref, r_const=r_const
    )

    # Calculate change in u (du_array)
    du_array = adjoint_jac_matrix @ u_atl_old

    return du_array


@njit(
    (types.Array(types.float64, 2, "C", readonly=False))(
        types.Array(types.float64, 2, "C", readonly=False),
        types.Array(types.float64, 1, "C", readonly=True),
        types.Array(types.float64, 2, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
        types.float64,
        types.float64,
    ),
    cache=cfg.NUMBA_CACHE,
)
def atl_runge_kutta4(
    u_atl_old: np.ndarray((sdim, 1)) = None,
    u_ref_old: np.ndarray = None,
    lorentz_matrix: np.ndarray = None,
    jacobian_matrix: np.ndarray = None,
    dt: float = 1,
    r_const: float = 28,
) -> Tuple[np.ndarray, np.ndarray]:

    # Calculate the k's of the non-linear model
    Y1 = u_ref_old
    k1 = dt * derivative_evaluator(u_old=Y1, lorentz_matrix=lorentz_matrix)
    Y2 = u_ref_old + 1 / 2 * k1
    k2 = dt * derivative_evaluator(u_old=Y2, lorentz_matrix=lorentz_matrix)
    Y3 = u_ref_old + 1 / 2 * k2
    k3 = dt * derivative_evaluator(u_old=Y3, lorentz_matrix=lorentz_matrix)
    Y4 = u_ref_old + k3

    # Calculate the l's of the tangent linear model
    l4 = atl_derivative_evaluator(
        u_atl_old=u_atl_old,
        u_ref=Y4,
        jacobian_matrix=jacobian_matrix,
        r_const=r_const,
    )
    l3 = atl_derivative_evaluator(
        u_atl_old=u_atl_old + 1 / 2 * dt * l4,
        u_ref=Y3,
        jacobian_matrix=jacobian_matrix,
        r_const=r_const,
    )
    l2 = atl_derivative_evaluator(
        u_atl_old=u_atl_old + 1 / 2 * dt * l3,
        u_ref=Y2,
        jacobian_matrix=jacobian_matrix,
        r_const=r_const,
    )
    l1 = atl_derivative_evaluator(
        u_atl_old=u_atl_old + dt * l2,
        u_ref=Y1,
        jacobian_matrix=jacobian_matrix,
        r_const=r_const,
    )

    # Update u_atl_old
    u_atl_old = u_atl_old + 1 / 6 * dt * (l1 + 2 * (l2 + l3) + l4)

    return u_atl_old
