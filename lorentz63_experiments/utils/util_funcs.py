import sys

sys.path.append("..")
from lorentz63_experiments.params.params import *


def setup_deriv_matrix(args):
    # Setup derivMatrix
    derivMatrix[0, 0] = -args["sigma"]
    derivMatrix[0, 1] = args["sigma"]
    derivMatrix[1, 0] = args["r_const"]
    derivMatrix[1, 1] = -1
    derivMatrix[2, 2] = -args["b_const"]

    return derivMatrix
