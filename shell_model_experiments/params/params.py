import sys

sys.path.append("..")
import config as cfg

# import numba as nb
import numpy as np
from general.utils.module_import.type_import import *
from numba import njit
from numba.core import types
from numba.experimental import structref

# from numba.experimental import jitclass, structref

# Define variables not suited for numba jitclass
dtype: type = np.complex128

# Define a StructRef.
# `structref.register` associates the type with the default data model.
# This will also install getters and setters to the fields of
# the StructRef.
@structref.register
class ParamsStructType(types.StructRef):
    def preprocess_fields(self, fields):
        # This method is called by the type constructor for additional
        # preprocessing on the fields.
        # Here, we don't want the struct to take Literal types.
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


# Define a Python type that can be use as a proxy to the StructRef
# allocated inside Numba. Users can construct the StructRef via
# the constructor for this type in python code and jit-code.
class ParamsStruct(structref.StructRefProxy):
    def __new__(
        cls,
        epsilon,
        lambda_const,
        dt,
        sample_rate,
        bd_size,
        n_forcing,
        u0,
        seeked_error_norm,
        sdim,
        tts,
        stt,
        factor2,
        factor3,
        k_vec_temp,
        pre_factor,
        hel_pre_factor,
        du_array,
        initial_k_vec,
    ):
        # Overriding the __new__ method is optional, doing so
        # allows Python code to use keyword arguments,
        # or add other customized behavior.
        # The default __new__ takes `*args`.
        # IMPORTANT: Users should not override __init__.
        return structref.StructRefProxy.__new__(
            cls,
            epsilon,
            lambda_const,
            dt,
            sample_rate,
            bd_size,
            n_forcing,
            u0,
            seeked_error_norm,
            sdim,
            tts,
            stt,
            factor2,
            factor3,
            k_vec_temp,
            pre_factor,
            hel_pre_factor,
            du_array,
            initial_k_vec,
        )

    # By default, the proxy type does not reflect the attributes or
    # methods to the Python side. It is up to users to define
    # these. (This may be automated in the future.)

    @property
    def epsilon(self):
        # To access a field, we can define a function that simply
        # return the field in jit-code.
        # The definition of struct_get_name is shown later.
        return struct_get_epsilon(self)

    @property
    def lambda_const(self):
        # The definition of struct_get_vector is shown later.
        return struct_get_lambda_const(self)

    @property
    def dt(self):
        # The definition of struct_get_vector is shown later.
        return struct_get_dt(self)

    @property
    def bd_size(self):
        # The definition of struct_get_bd_size is shown later.
        return struct_get_bd_size(self)

    @property
    def n_forcing(self):
        # The definition of struct_get_n_forcing is shown later.
        return struct_get_n_forcing(self)

    @property
    def u0(self):
        # The definition of struct_get_u0 is shown later.
        return struct_get_u0(self)

    @property
    def seeked_error_norm(self):
        # The definition of struct_get_seeked_error_norm is shown later.
        return struct_get_seeked_error_norm(self)

    @property
    def sdim(self):
        # The definition of struct_get_vector is shown later.
        return struct_get_sdim(self)

    @property
    def sample_rate(self):
        # The definition of struct_get_vector is shown later.
        return struct_get_sample_rate(self)

    @property
    def tts(self):
        # The definition of struct_get_vector is shown later.
        return struct_get_tts(self)

    @property
    def stt(self):
        # The definition of struct_get_stt is shown later.
        return struct_get_stt(self)

    @property
    def factor2(self):
        # The definition of struct_get_factor2 is shown later.
        return struct_get_factor2(self)

    @property
    def factor3(self):
        # The definition of struct_get_factor3 is shown later.
        return struct_get_factor3(self)

    @property
    def k_vec_temp(self):
        # The definition of struct_get_vector is shown later.
        return struct_get_k_vec_temp(self)

    @property
    def pre_factor(self):
        # The definition of struct_pre_factor is shown later.
        return struct_get_pre_factor(self)

    @property
    def hel_pre_factor(self):
        # The definition of struct_get_hel_pre_factor is shown later.
        return struct_get_hel_pre_factor(self)

    @property
    def du_array(self):
        # The definition of strucdu_array is shown later.
        return struct_get_du_array(self)

    @property
    def initial_k_vec(self):
        # The definition of struct_getinitial_k_vec is shown later.
        return struct_get_initial_k_vec(self)


# Define python wrappers for the struct


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_epsilon(self):
    # In jit-code, the StructRef's attribute is exposed via
    # structref.register
    return self.epsilon


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_lambda_const(self):
    return self.lambda_const


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_dt(self):
    return self.dt


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_bd_size(self):
    return self.bd_size


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_n_forcing(self):
    return self.n_forcing


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_u0(self):
    return self.u0


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_seeked_error_norm(self):
    return self.seeked_error_norm


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_sdim(self):
    return self.sdim


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_sample_rate(self):
    return self.sample_rate


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_tts(self):
    return int(round(self.sample_rate / self.dt))


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_stt(self):
    return int(round(self.dt / self.sample_rate))


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_factor2(self):
    return -self.epsilon / self.lambda_const


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_factor3(self):
    return (1 - self.epsilon) / self.lambda_const ** 2


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_k_vec_temp(self):
    return self.k_vec_temp


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_pre_factor(self):
    return self.pre_factor


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_hel_pre_factor(self):
    return self.hel_pre_factor


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_du_array(self):
    return self.du_array


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_initial_k_vec(self):
    return self.initial_k_vec


# This associates the proxy with ParamsStructType for the given set of
# fields. Notice how we are not contraining the type of each field.
# Field types remain generic.
structref.define_proxy(
    ParamsStruct,
    ParamsStructType,
    [
        "epsilon",
        "lambda_const",
        "dt",
        "sample_rate",
        "bd_size",
        "n_forcing",
        "u0",
        "seeked_error_norm",
        "sdim",
        "tts",
        "stt",
        "factor2",
        "factor3",
        "k_vec_temp",
        "pre_factor",
        "hel_pre_factor",
        "du_array",
        "initial_k_vec",
    ],
)

PAR = ParamsStruct(
    epsilon=0.5,
    lambda_const=2,
    dt=1e-7,
    sdim=20,
    sample_rate=1 / 1000,
    tts=0,
    bd_size=2,
    n_forcing=0,
    u0=1,
    seeked_error_norm=1e-4,
    stt=0,
    factor2=0,
    factor3=0,
    k_vec_temp=np.empty(0, dtype=np.float64),
    pre_factor=np.empty(0, dtype=np.complex128),
    hel_pre_factor=np.empty(0, dtype=np.float64),
    du_array=np.empty(0, dtype=np.complex128),
    initial_k_vec=np.empty(0, dtype=np.float64),
)

# @njit(cache=cfg.NUMBA_CACHE)
# def set_params(struct: ParamsStructType, key, value):
#     # for key, value in fields.items():
#     struct.__setattr__(key, value)


# @njit((structref.StructRefProxy, types.int32), cache=cfg.NUMBA_CACHE)
# def use_alice(struct, n_runs: int):
#     for _ in range(n_runs):
#         s = struct.sdim

# Define variables not suited for numba jitclass
# u_slice: slice = slice(PAR.bd_size, -PAR.bd_size, 1)


# @jitclass(
#     [
#         ("epsilon", nb.types.float64),
#         ("lambda_const", nb.types.float64),
#         ("dt", nb.types.float64),
#         ("sample_rate", nb.types.float64),
#         ("bd_size", nb.types.int16),
#         ("n_forcing", nb.types.int16),
#         ("u0", nb.types.float64),
#         ("seeked_error_norm", nb.types.float64),
#         ("sdim", nb.types.int32),
#         ("tts", nb.types.float64),
#         ("stt", nb.types.float64),
#         ("factor2", nb.types.float64),
#         ("factor3", nb.types.float64),
#         ("k_vec_temp", nb.types.Array(nb.types.float64, 1, "C", readonly=True)),
#         ("pre_factor", nb.types.Array(nb.types.complex128, 1, "C", readonly=True)),
#         ("hel_pre_factor", nb.types.Array(nb.types.float64, 1, "C", readonly=True)),
#         ("du_array", nb.types.Array(nb.types.complex128, 1, "C", readonly=False)),
#         ("initial_k_vec", nb.types.Array(nb.types.float64, 1, "C", readonly=True)),
#     ],
# )
# class Params:
#     def __init__(self):
#         self.epsilon: float = 0.5
#         self.lambda_const: float = 2
#         self.dt: float = 1e-7
#         self.sample_rate: float = 1 / 1000
#         self.bd_size: int = 2
#         self.n_forcing: int = 0
#         self.u0: float = 1
#         self.seeked_error_norm: float = 1e-4
#         self.sdim: int = 20

#         # Define conversion factors
#         self.tts: float = self.sample_rate / self.dt
#         self.stt: float = self.dt / self.sample_rate

#         # Define factors for derivative calculation
#         self.factor2: float = -self.epsilon / self.lambda_const
#         self.factor3: float = (1 - self.epsilon) / self.lambda_const ** 2

#         # Define arrays
#         self.k_vec_temp: np.ndarray = np.array(
#             [self.lambda_const ** (n + 1) for n in range(self.sdim)],
#             dtype=np.float64,
#         )
#         self.pre_factor: np.ndarray = 1j * self.k_vec_temp
#         self.hel_pre_factor: np.ndarray = (-1) ** (
#             np.arange(1, self.sdim + 1)
#         ) * self.k_vec_temp
#         self.du_array: np.ndarray = np.zeros(self.sdim + 2 * self.bd_size, dtype=dtype)
#         self.initial_k_vec: np.ndarray = self.k_vec_temp ** (-1 / 3)

#     def initialise_sdim_arrays(self):

#         # Define k vector indices array
#         self.k_vec_temp = np.array(
#             [self.lambda_const ** (n + 1) for n in range(self.sdim)],
#             dtype=np.float64,
#         )

#         # Define pre_factor for derivative calculation
#         self.pre_factor = 1j * self.k_vec_temp

#         # Helicity pre factor
#         self.hel_pre_factor = (-1) ** (np.arange(1, self.sdim + 1)) * self.k_vec_temp

#         # Define du array to store derivative
#         self.du_array = np.zeros(self.sdim + 2 * self.bd_size, dtype=dtype)

#         # Calculate initial k and u profile. Put in zeros at the boundaries
#         self.initial_k_vec = self.k_vec_temp ** (-1 / 3)


# @nb.jit(cache=cfg.NUMBA_CACHE)
# def wrapper():
#     PAR = Params()
#     return PAR


# PAR = wrapper()


# print(hex(id(PAR.pre_factor)), PAR.pre_factor.size)
# print(hex(id(PAR.pre_factor)), PAR.pre_factor.size)
# PAR.sdim = 30
# PAR.initialise_sdim_arrays()
# print(hex(id(PAR.pre_factor)), PAR.pre_factor.size)
# print(hex(id(PAR.pre_factor)), PAR.pre_factor.size)


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
