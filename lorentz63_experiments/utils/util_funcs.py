import sys

sys.path.append("..")
from lorentz63_experiments.params.params import *


def setup_lorentz_matrix(args):
    # Setup lorentz_matrix
    lorentz_matrix[0, 0] = -args["sigma"]
    lorentz_matrix[0, 1] = args["sigma"]
    lorentz_matrix[1, 0] = args["r_const"]
    lorentz_matrix[1, 1] = -1
    lorentz_matrix[2, 2] = -args["b_const"]

    return lorentz_matrix
