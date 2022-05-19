from colorama import init as colorama_init
import general.plotting.plot_config as plt_config
from general.params.params import GlobalParams
import general.utils.licence_utils.model_licence_utils as md_license_ut
import general.utils.licence_utils.exp_licence_utils as exp_licence_ut
from general.params.model_licences import Models


# Init colorama
colorama_init()

# Get model
MODEL = md_license_ut.detect_model_in_use()

# Initialise licence
def init_licence():
    global LICENCE
    LICENCE = exp_licence_ut.detect_exp_licence()


# Get global params
GLOBAL_PARAMS = GlobalParams()

if MODEL == Models.LORENTZ63:
    GLOBAL_PARAMS.record_max_time = 10000
else:
    GLOBAL_PARAMS.record_max_time = 30

# Set plotting config
# plt_config.latex_plot_settings()

# Other general configurations
NUMBA_CACHE = True
NUMBA_ON = True
