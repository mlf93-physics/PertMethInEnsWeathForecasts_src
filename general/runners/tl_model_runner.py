import sys

sys.path.append("..")
import math
from pyinstrument import Profiler
import numpy as np
from lorentz63_experiments.params.params import *
import general.utils.argument_parsers as a_parsers
import lorentz63_experiments.lorentz63_model.tl_lorentz63 as l63_tl_model

from general.params.model_licences import Models
from config import MODEL, LICENCE, GLOBAL_PARAMS

profiler = Profiler()

# Get parameters for model
# if MODEL == Models.SHELL_MODEL:
#     params = sh_params
# elif MODEL == Models.LORENTZ63:
#     params = l63_params

# Set global params
GLOBAL_PARAMS.ref_run = False


def main(args):
    # Set arguments for ref import
    args["ref_start_time"] = args["start_times"][0]
    args["ref_end_time"] = args["start_times"][0] + args["time_to_run"]

    if MODEL == Models.SHELL_MODEL:
        print("No TL model for the shell model yet")
    elif MODEL == Models.LORENTZ63:
        l63_tl_model.main(args)


if __name__ == "__main__":
    # Get arguments
    rel_ref_arg_setup = a_parsers.RelReferenceArgSetup()
    rel_ref_arg_setup.setup_parser()
    ref_arg_setup = a_parsers.ReferenceAnalysisArgParser()
    ref_arg_setup.setup_parser()
    args = ref_arg_setup.args

    # Add/edit arguments
    args["Nt"] = int(args["time_to_run"] / dt)

    main(args)
