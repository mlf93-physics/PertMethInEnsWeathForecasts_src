import sys

sys.path.append("..")
from pyinstrument import Profiler
from lorentz63_experiments.params.params import *
import general.utils.argument_parsers as a_parsers
import lorentz63_experiments.lorentz63_model.tl_lorentz63 as l63_tl_model
import general.utils.user_interface as g_ui
import general.utils.running.runner_utils as r_utils
from general.params.model_licences import Models
import config as cfg

profiler = Profiler()

# Set global params
cfg.GLOBAL_PARAMS.ref_run = False


def main(args):
    # Set arguments for ref import
    args["ref_start_time"] = args["start_times"][0]
    args["ref_end_time"] = args["start_times"][0] + args["time_to_run"]

    if cfg.MODEL == Models.SHELL_MODEL:
        print("No TL model for the shell model yet")
    elif cfg.MODEL == Models.LORENTZ63:
        l63_tl_model.main(args)


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    rel_ref_arg_setup = a_parsers.RelReferenceArgSetup()
    rel_ref_arg_setup.setup_parser()
    ref_arg_setup = a_parsers.ReferenceAnalysisArgParser()
    ref_arg_setup.setup_parser()
    args = ref_arg_setup.args

    g_ui.confirm_run_setup(args)
    r_utils.adjust_run_setup(args)

    # Add/edit arguments
    args["Nt"] = int(args["time_to_run"] / dt)

    main(args)
