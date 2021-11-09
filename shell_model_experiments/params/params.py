import math
import numpy as np

#### Initialise sabra model constants ####
epsilon = 0.5
lambda_const = 2
dt = 1e-7
sample_rate = 1 / 1000
bd_size = 2
n_forcing = 0
u0 = 1

# Define some conversion shortcuts
tts = sample_rate / dt
stt = dt / sample_rate

# Define types
dtype = np.complex128

# Define factors to be used in the derivative calculation
factor2 = -epsilon / lambda_const
factor3 = (1 - epsilon) / lambda_const ** 2


# Define u_slice
u_slice = np.s_[bd_size:-bd_size:1]


#### Initialise Lyaponov exponent estimator constants ####
seeked_error_norm = 1e-4


def initiate_sdim_arrays(local_sdim: int):
    """Initiate arrays based on sdim

    Parameters
    ----------
    local_sdim : int
        The dimension of the system
    """

    #### Initialise sabra model arrays ####
    global sdim, k_vec_temp, pre_factor, hel_pre_factor, du_array, initial_k_vec

    # Define the global sdim
    sdim = local_sdim
    # Define k vector indices
    k_vec_temp = np.array(
        [lambda_const ** (n + 1) for n in range(sdim)], dtype=np.float64
    )
    pre_factor = 1j * k_vec_temp

    # Helicity pre factor
    hel_pre_factor = (-1) ** (np.arange(1, sdim + 1)) * k_vec_temp

    # Define du array to store derivative
    du_array = np.zeros(sdim + 2 * bd_size, dtype=dtype)

    # Calculate initial k and u profile. Put in zeros at the boundaries
    initial_k_vec = k_vec_temp ** (-1 / 3)


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
