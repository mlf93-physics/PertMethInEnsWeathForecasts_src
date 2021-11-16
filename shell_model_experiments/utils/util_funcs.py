import sys

sys.path.append("..")
import math
import numpy as np
import numba as nb
from shell_model_experiments.params.params import PAR, ParamsStructType
import shell_model_experiments.utils.special_params as sparams
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
    ny = (forcing / (PAR.lambda_const ** (2 * ny_n * (diff_exponent - 2 / 3)))) ** (
        1 / 2
    )

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
        / (2 * (diff_exponent - 2 / 3) * math.log10(PAR.lambda_const))
    )

    return ny_n


@nb.njit(
    (nb.types.Array(nb.types.complex128, 1, "C", readonly=False))(
        nb.types.Array(nb.types.complex128, 1, "C", readonly=False),
        nb.typeof(PAR),
        nb.types.float64,
        nb.types.float64,
    ),
    cache=cfg.NUMBA_CACHE,
)
def normal_diffusion(u_old, PAR, ny, diff_exponent):
    # Solve linear diffusive term explicitly
    u_old[PAR.bd_size : -PAR.bd_size] = u_old[PAR.bd_size : -PAR.bd_size] * np.exp(
        -ny * PAR.k_vec_temp ** diff_exponent * PAR.dt
    )
    return u_old


@nb.njit(
    (nb.types.Array(nb.types.complex128, 1, "C", readonly=False))(
        nb.types.Array(nb.types.complex128, 1, "C", readonly=False),
        nb.typeof(PAR),
        nb.types.float64,
        nb.types.float64,
    ),
    cache=cfg.NUMBA_CACHE,
)
def infinit_hyper_diffusion(u_old, PAR, ny, diff_exponent):
    # Use infinit hyperdiffusion
    u_old[-(PAR.bd_size + 1)] = 0
    return u_old


@nb.njit((nb.types.none)(nb.typeof(PAR)), cache=cfg.NUMBA_CACHE)
def update_arrays(struct: ParamsStructType):
    """Update arrays based on other parameters

    Parameters
    ----------
    struct : ParamsStructType
        The struct holding parameters and arrays

    """
    struct.k_vec_temp = np.array(
        [struct.lambda_const ** (n + 1) for n in range(struct.sdim)],
        dtype=np.float64,
    )
    struct.pre_factor = 1j * struct.k_vec_temp
    struct.hel_pre_factor = (-1) ** (np.arange(1, struct.sdim + 1)) * struct.k_vec_temp
    struct.du_array = np.zeros(struct.sdim + 2 * struct.bd_size, dtype=sparams.dtype)
    struct.initial_k_vec = struct.k_vec_temp ** (-1 / 3)


@nb.njit(
    [
        (nb.types.none)(nb.typeof(PAR), nb.types.int32),
        (nb.types.none)(nb.typeof(PAR), nb.types.Omitted(None)),
    ],
    cache=cfg.NUMBA_CACHE,
)
def set_params(struct: ParamsStructType, sdim: int = None):
    """Set parameters in param struct

    Parameters
    ----------
    struct : ParamsStructType
        The parameter struct
    sdim : int, optional
        The dimension parameter, by default None
    """
    if sdim is not None:
        struct.sdim = sdim


@nb.njit(
    [
        (nb.types.none)(nb.typeof(PAR), nb.types.int32),
        (nb.types.none)(nb.typeof(PAR), nb.types.Omitted(None)),
    ],
    cache=cfg.NUMBA_CACHE,
)
def update_dependent_params(struct: ParamsStructType, sdim: int = None):
    """Update parameters based on other parameters and specific params given
    as input arguments

    Parameters
    ----------
    struct : ParamsStructType
        The struct holding parameters and arrays
    """

    # Update parameters based on other parameters
    struct.tts = int(round(struct.sample_rate / struct.dt))
    struct.stt = struct.dt / struct.sample_rate
    struct.factor2 = -struct.epsilon / struct.lambda_const
    struct.factor3 = (1 - struct.epsilon) / struct.lambda_const ** 2

    if sdim is not None:
        set_params(struct, sdim=sdim)


def format_params_to_string():
    """Run through all attributes of PAR and format attr, value pair into string

    Returns
    -------
    str
        The resulting string of attr, value pairs
    """

    attr_list: list = []

    for attr in dir(PAR):
        if (
            not attr.startswith("__")
            and not attr.startswith("_")
            and not callable(getattr(PAR, attr))
            and not isinstance(getattr(PAR, attr), np.ndarray)
        ):
            attr_list.append((attr, PAR.__getattribute__(attr)))

    string: str = ", ".join(map(lambda item: f"{item[0]}={item[1]}", attr_list))

    return string
