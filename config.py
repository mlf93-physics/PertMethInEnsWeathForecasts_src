__all__ = ["MODEL"]

import os
import pathlib as pl
from general.params.model_licences import Models

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


MODEL = detect_model_in_use()
