__all__ = ["MODEL", "LICENCE", "NUMBA_CACHE"]

import os
import pathlib as pl
from general.params.model_licences import Models
from general.params.experiment_licences import Experiments
from general.params.params import GlobalParams
import general.utils.user_interface as g_ui

models = Models()


def detect_model_in_use():
    cwd = str(pl.Path(os.getcwd()))

    if "lorentz63" in cwd:
        model = models.LORENTZ63
    elif "shell_model" in cwd:
        model = models.SHELL_MODEL
    else:
        raise NameError("Cannot detect which model is in use")

    return model


def confirm_run_setup():
    print("CONFIRM SETUP TO RUN:")
    print(f"Licence: {LICENCE}")
    print(f"Numba cache: {NUMBA_CACHE}")
    print("\n")

    confirm = g_ui.ask_user("Please confirm the current setup to run")

    if not confirm:
        exit()


# Get model
MODEL = detect_model_in_use()
print(f"\nRunning with model {MODEL}\n")

# Set experiment license
exp = Experiments()
LICENCE = exp.LYAPUNOV_VECTORS

# Get global params
GLOBAL_PARAMS = GlobalParams()

# Other general configurations
NUMBA_CACHE = True
# Disable stdout
# sys.stdout = open(os.devnull, "w")

confirm_run_setup()
