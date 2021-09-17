__all__ = ["MODEL", "LICENCE"]

import os
import pathlib as pl
from general.params.model_licences import Models
from general.params.experiment_licences import Experiments

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


# Get model
MODEL = detect_model_in_use()

# Set experiment license
exp = Experiments()
LICENCE = exp.NORMAL_PERTURBATION
