import sys

sys.path.append("..")
import math
import numpy as np
import numba as nb
from numba.experimental import jitclass
from general.utils.module_import.type_import import *


@jitclass(
    [
        ("epsilon", nb.types.float64),
        ("lambda_const", nb.types.float64),
        ("dt", nb.types.float64),
        ("sample_rate", nb.types.float64),
        ("bd_size", nb.types.int16),
        ("n_forcing", nb.types.int16),
        ("u0", nb.types.float64),
        ("seeked_error_norm", nb.types.float64),
        ("sdim", nb.types.int32),
        ("tts", nb.types.float64),
        ("stt", nb.types.float64),
        ("factor2", nb.types.float64),
        ("factor3", nb.types.float64),
        ("k_vec_temp", nb.types.Array(nb.types.float64, 1, "C", readonly=True)),
        ("pre_factor", nb.types.Array(nb.types.complex128, 1, "C", readonly=True)),
        ("hel_pre_factor", nb.types.Array(nb.types.float64, 1, "C", readonly=True)),
        ("du_array", nb.types.Array(nb.types.complex128, 1, "C", readonly=False)),
        ("initial_k_vec", nb.types.Array(nb.types.float64, 1, "C", readonly=True)),
    ],
)
class Params:
    def __init__(self):
        self.epsilon: float = 0.5
        self.lambda_const: float = 2
        self.dt: float = 1e-7
        self.sample_rate: float = 1 / 1000
        self.bd_size: int = 2
        self.n_forcing: int = 0
        self.u0: float = 1
        self.seeked_error_norm: float = 1e-4
        self.sdim: int = 20

        # Define conversion factors
        self.tts: float = self.sample_rate / self.dt
        self.stt: float = self.dt / self.sample_rate

        # Define factors for derivative calculation
        self.factor2: float = -self.epsilon / self.lambda_const
        self.factor3: float = (1 - self.epsilon) / self.lambda_const ** 2

        # Define arrays
        self.k_vec_temp: np.ndarray = np.array(
            [self.lambda_const ** (n + 1) for n in range(self.sdim)],
            dtype=np.float64,
        )
        self.pre_factor: np.ndarray = 1j * self.k_vec_temp
        self.hel_pre_factor: np.ndarray = (-1) ** (
            np.arange(1, self.sdim + 1)
        ) * self.k_vec_temp
        self.du_array: np.ndarray = np.zeros(
            self.sdim + 2 * self.bd_size, dtype=np.complex128
        )
        self.initial_k_vec: np.ndarray = self.k_vec_temp ** (-1 / 3)

    def initialise_arrays(self):

        # Define k vector indices array
        self.k_vec_temp = np.array(
            [self.lambda_const ** (n + 1) for n in range(self.sdim)],
            dtype=np.float64,
        )

        # Define pre_factor for derivative calculation
        self.pre_factor = 1j * self.k_vec_temp

        # Helicity pre factor
        self.hel_pre_factor = (-1) ** (np.arange(1, self.sdim + 1)) * self.k_vec_temp

        # Define du array to store derivative
        self.du_array = np.zeros(self.sdim + 2 * self.bd_size, dtype=np.complex128)

        # Calculate initial k and u profile. Put in zeros at the boundaries
        self.initial_k_vec = self.k_vec_temp ** (-1 / 3)


PAR = Params()

print(hex(id(PAR.pre_factor)), PAR.pre_factor.size)
print(hex(id(PAR.pre_factor)), PAR.pre_factor.size)
PAR.sdim = 30
PAR.initialise_arrays()
print(hex(id(PAR.pre_factor)), PAR.pre_factor.size)
print(hex(id(PAR.pre_factor)), PAR.pre_factor.size)

# Define variables not suited for numba jitclass
u_slice: slice = slice(PAR.bd_size, -PAR.bd_size, 1)
dtype: type = np.complex128

# # array1 = np.arange(20)

# s1 = s.k_vec_temp
# print("size", s1.size)
# s.sdim = 30
# s2 = s.k_vec_temp
# print("size", s2.size)
# print(s1 is s2)
# s3 = s.k_vec_temp
# print("size", s3.size)
# print(s3 is s2)
# # print(hex(id(s.k_vec_temp)))

# # print(hex(id(s.k_vec_temp)))

# #### Initialise sabra model constants ####
# epsilon = 0.5
# lambda_const = 2
# dt = 1e-7
# sample_rate = 1 / 1000
# bd_size = 2
# n_forcing = 0
# u0 = 1

# # Define some conversion shortcuts
# tts = sample_rate / dt
# stt = dt / sample_rate

# # Define types
# dtype = np.complex128

# # Define factors to be used in the derivative calculation
# factor2 = -epsilon / lambda_const
# factor3 = (1 - epsilon) / lambda_const ** 2


# # Define u_slice
# u_slice = np.s_[bd_size:-bd_size:1]

# #### Initialise Lyaponov exponent estimator constants ####
# seeked_error_norm = 1e-4


# def initiate_sdim_arrays(local_sdim: int):
#     """Initiate arrays based on sdim

#     Parameters
#     ----------
#     local_sdim : int
#         The dimension of the system
#     """

#     #### Initialise sabra model arrays ####
#     global sdim, k_vec_temp, pre_factor, hel_pre_factor, du_array, initial_k_vec

#     # Define the global sdim
#     sdim = local_sdim
#     # Define k vector indices
#     k_vec_temp = np.array(
#         [lambda_const ** (n + 1) for n in range(sdim)], dtype=np.float64
#     )
#     pre_factor = 1j * k_vec_temp

#     # Helicity pre factor
#     hel_pre_factor = (-1) ** (np.arange(1, sdim + 1)) * k_vec_temp

#     # Define du array to store derivative
#     du_array = np.zeros(sdim + 2 * bd_size, dtype=dtype)

#     # Calculate initial k and u profile. Put in zeros at the boundaries
#     initial_k_vec = k_vec_temp ** (-1 / 3)
