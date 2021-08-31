import numpy as np
from math import ceil

#### Initialise sabra model constants ####
epsilon = 0.5
lambda_const = 2
dt = 1e-7
sample_rate = 1/1000
n_k_vec = 20
bd_size = 2
n_forcing = 0
u0 = 1

# Define factors to be used in the derivative calculation
factor2 = - epsilon/lambda_const
factor3 = (1 - epsilon)/lambda_const**2

#### Initialise sabra model arrays ####
# Define k vector indices
k_vec_temp = np.array([lambda_const**(n + 1) for n in range(n_k_vec)],
    dtype=np.int64)
pre_factor = 1j*k_vec_temp
# Define du array to store derivative
du_array = np.zeros(n_k_vec + 2*bd_size, dtype=np.complex128)
# # Define data out array to store what should be saved.
# data_out = np.zeros((int(Nt*sample_rate), n_k_vec + 1), dtype=np.complex128)

# Calculate initial k and u profile. Put in zeros at the boundaries
initial_k_vec = k_vec_temp**(-1/3)

#### Initialise Lyaponov exponent estimator constants ####
seeked_error_norm = 1e-14
