import numpy as np
from shell_model_experiments.params.params import PAR
from shell_model_experiments.params.params import ParamsStructType

# Define variables not suited for numba structref
dtype: type = np.complex128
u_slice = slice(PAR.bd_size, -PAR.bd_size, 1)
