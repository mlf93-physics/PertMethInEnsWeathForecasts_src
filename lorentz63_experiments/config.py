__all__ = ["LICENCE"]
import sys

sys.path.append("..")
from general.params.experiment_licences import Experiments

exp = Experiments()

LICENCE = exp.LORENTZ_BLOCK
