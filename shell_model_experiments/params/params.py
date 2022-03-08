"""Define parameters to be used in sabra model

Refer to https://numba.readthedocs.io/en/0.51.2/extending/high-level.html
for documentation of structrefs

"""

import sys

sys.path.append("..")
import config as cfg
import numpy as np
from general.utils.module_import.type_import import *
from numba import njit
from numba.core import types
from numba.experimental import structref

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
        # ny,
        epsilon,
        lambda_const,
        dt,
        sample_rate,
        bd_size,
        forcing,
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
            # ny,
            epsilon,
            lambda_const,
            dt,
            sample_rate,
            bd_size,
            forcing,
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

    # @property
    # def ny(self):
    #     # The definition of lambda_const is shown later.
    #     return struct_get_ny(self)

    @property
    def epsilon(self):
        # To access a field, we can define a function that simply
        # return the field in jit-code.
        # The definition of struct_get_name is shown later.
        return struct_get_epsilon(self)

    @property
    def lambda_const(self):
        # The definition of lambda_const is shown later.
        return struct_get_lambda_const(self)

    @property
    def dt(self):
        # The definition of dt is shown later.
        return struct_get_dt(self)

    @property
    def bd_size(self):
        # The definition of struct_get_bd_size is shown later.
        return struct_get_bd_size(self)

    @property
    def forcing(self):
        # The definition of struct_get_forcing is shown later.
        return struct_get_forcing(self)

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
        # The definition of sdim is shown later.
        return struct_get_sdim(self)

    @property
    def sample_rate(self):
        # The definition of sample_rate is shown later.
        return struct_get_sample_rate(self)

    @property
    def tts(self):
        # The definition of tts is shown later.
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
        # The definition of k_vec_temp is shown later.
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
# In jit-code, the StructRef's attribute is exposed via
# structref.register
# @njit(cache=cfg.NUMBA_CACHE)
# def struct_get_ny(self):
#     return self.ny


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_epsilon(self):
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
def struct_get_forcing(self):
    return self.forcing


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
    return self.tts


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_stt(self):
    return self.stt


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_factor2(self):
    return self.factor2


@njit(cache=cfg.NUMBA_CACHE)
def struct_get_factor3(self):
    return self.factor3


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
        # "ny",
        "epsilon",
        "lambda_const",
        "dt",
        "sample_rate",
        "bd_size",
        "forcing",
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

# Instanciate ParamsStruct
# Remember to indicate variable type through default value, e.g. 0.0 for float
# instead of 0 (if set to 0 -> variable will be integer)
PAR = ParamsStruct(
    # ny=1e-8,
    epsilon=0.5,
    lambda_const=2.0,
    dt=1e-5,
    sdim=20,
    sample_rate=0.1,
    bd_size=2,
    forcing=1,
    n_forcing=0,
    u0=1,
    seeked_error_norm=1e-4,
    tts=0,
    stt=0.0,
    factor2=0.0,
    factor3=0.0,
    k_vec_temp=np.empty(0, dtype=np.float64),
    pre_factor=np.empty(0, dtype=np.complex128),
    hel_pre_factor=np.empty(0, dtype=np.float64),
    du_array=np.empty(0, dtype=np.complex128),
    initial_k_vec=np.empty(0, dtype=np.float64),
)
