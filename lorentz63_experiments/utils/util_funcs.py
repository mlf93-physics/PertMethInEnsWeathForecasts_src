import sys

sys.path.append("..")
from lorentz63_experiments.params.params import *


def setup_deriv_matrix(args):
    # Setup deriv_matrix
    deriv_matrix[0, 0] = -args["sigma"]
    deriv_matrix[0, 1] = args["sigma"]
    deriv_matrix[1, 0] = args["r_const"]
    deriv_matrix[1, 1] = -1
    deriv_matrix[2, 2] = -args["b_const"]

    return deriv_matrix
