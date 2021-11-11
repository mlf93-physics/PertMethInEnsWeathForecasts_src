import sys

sys.path.append("..")
import math
from numba import njit, types
from shell_model_experiments.params.params import *
import config as cfg


def ny_from_ny_n_and_forcing(forcing, ny_n, diff_exponent):
    """Util func to calculate ny from forcing and ny_n

    Parameters
    ----------
    forcing : float
        The applied forcing
    ny_n : int
        The shell at which to apply the diffusion

    Returns
    -------
    float
        The viscosity
    """
    ny = (forcing / (lambda_const ** (2 * ny_n * (diff_exponent - 2 / 3)))) ** (1 / 2)

    return ny


def ny_n_from_ny_and_forcing(forcing, ny, diff_exponent):
    """Util func to calculate ny_n from forcing and ny

    Parameters
    ----------
    forcing : float
        The applied forcing
    ny : float
        The viscosity

    Returns
    -------
    int
        The shell at which to apply the viscosity
    """
    ny_n = int(
        math.log10(forcing / (ny ** 2))
        / (2 * (diff_exponent - 2 / 3) * math.log10(lambda_const))
    )

    return ny_n


@njit(
    (types.Array(types.complex128, 1, "C", readonly=False))(
        types.Array(types.complex128, 1, "C", readonly=False),
        types.float64,
        types.float64,
    ),
    cache=cfg.NUMBA_CACHE,
)
def normal_diffusion(u_old, ny, diff_exponent):
    # Solve linear diffusive term explicitly
    u_old[bd_size:-bd_size] = u_old[bd_size:-bd_size] * np.exp(
        -ny * k_vec_temp ** diff_exponent * dt
    )
    return u_old


@njit(
    (types.Array(types.complex128, 1, "C", readonly=False))(
        types.Array(types.complex128, 1, "C", readonly=False),
        types.float64,
        types.float64,
    ),
    cache=cfg.NUMBA_CACHE,
)
def infinit_hyper_diffusion(u_old, ny, diff_exponent):
    # Use infinit hyperdiffusion
    u_old[-(bd_size + 1)] = 0
    return u_old
