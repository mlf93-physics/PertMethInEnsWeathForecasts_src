import sys

sys.path.append("..")
import math
from shell_model_experiments.params.params import *


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
