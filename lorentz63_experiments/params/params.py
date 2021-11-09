import numpy as np

# Parameters for the Lorentz63 model
sample_rate = 1  # sample rate of saved data
dt = 0.01  # the timestep
bd_size = 0

# Parameters for perturbations
# For the Lorentz63 model, some perturbations can settle down to reference if
# norm is ~ 1e-14
seeked_error_norm = 1e-2

# Define some conversion shortcuts
tts = sample_rate / dt
stt = dt / sample_rate

# Define types
dtype = np.float64

# Define u_slice
u_slice = np.s_[0::]


def initiate_sdim_arrays(sdim_local: int):
    """Initiate the arrays depending on sdim

    Parameters
    ----------
    sdim_local : int
        The dimension of the system
    """
    global sdim, du_array, deriv_matrix
    # Make sdim global
    sdim = sdim_local
    # Arrays
    du_array = np.zeros(sdim, dtype=np.float64)
    deriv_matrix = np.zeros((sdim, sdim))
