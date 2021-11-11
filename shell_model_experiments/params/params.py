import math
import numpy as np

#### Initialise sabra model constants ####
epsilon = 0.5
lambda_const = 2
dt = 1e-7
sample_rate = 1 / 1000
sdim = 20
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

#### Initialise sabra model arrays ####
# Define k vector indices
k_vec_temp = np.array([lambda_const ** (n + 1) for n in range(sdim)], dtype=np.float64)
pre_factor = 1j * k_vec_temp

# Helicity pre factor
hel_pre_factor = (-1) ** (np.arange(1, sdim + 1)) * k_vec_temp
# Define du array to store derivative
du_array = np.zeros(sdim + 2 * bd_size, dtype=dtype)
# Define u_slice
u_slice = np.s_[bd_size:-bd_size:1]

# Calculate initial k and u profile. Put in zeros at the boundaries
initial_k_vec = k_vec_temp ** (-1 / 3)

#### Initialise Lyaponov exponent estimator constants ####
seeked_error_norm = 1e-4
