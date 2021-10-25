__all__ = ["MODEL", "LICENCE", "NUMBA_CACHE"]

import os
import sys
import pathlib as pl
from general.params.params import GlobalParams
import general.utils.licence_utils.model_licence_utils as md_license_ut
import general.utils.licence_utils.exp_licence_utils as exp_licence_ut
import general.plotting.plot_config as plt_config

# Get model
MODEL = md_license_ut.detect_model_in_use()

# Get licence
LICENCE = exp_licence_ut.detect_exp_licence()

# Get global params
GLOBAL_PARAMS = GlobalParams()

# Set plotting config
# plt_config.latex_plot_settings()

# Other general configurations
NUMBA_CACHE = True
NUMBA_ON = True
# Disable stdout
# sys.stdout = open(os.devnull, "w")
