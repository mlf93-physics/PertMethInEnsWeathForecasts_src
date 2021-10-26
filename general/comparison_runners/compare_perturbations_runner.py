"""Runs a complete set of perturbations based on several different perturbation
methods [rd, bv, bv-eof, ...]. Used to compare the different perturbation methods

Example
-------


"""

import sys

sys.path.append("..")
import pathlib as pl
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
import general.utils.experiments.exp_utils as exp_utils
import general.utils.user_interface as g_ui
import general.utils.argument_parsers as a_parsers
from general.runners.breed_vector_runner import main as bv_runner
from general.params.experiment_licences import Experiments as exp
from general.params.model_licences import Models
import config as cfg

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    params = sh_params
elif cfg.MODEL == Models.LORENTZ63:
    params = l63_params

# Set global params
cfg.GLOBAL_PARAMS.ref_run = False


def generate_bvs(args: dict, exp_setup: dict):
    """Generate the BVs according to the exp setup

    Parameters
    ----------
    args : dict
        Run-time arguments
    exp_setup : dict
        Experiment setup
    """
    cfg.LICENCE = exp.BREEDING_VECTORS

    # Set licence
    # Get subset of experiment setup
    local_exp_setup = exp_setup["general"] | exp_setup["bv_setup"]

    bv_runner(args, local_exp_setup)


def generate_vectors(args: dict, exp_setup: dict):
    args["save_last_pert"] = True

    generate_bvs(args, exp_setup)


def main(args: dict):
    """Main runner of the comparison

    Parameters
    ----------
    args : dict
        Run-time arguments
    """
    # Get experiment setup
    exp_file_path: pl.Path = pl.Path(
        "./params/experiment_setups/compare_pert_experiment_setups.json"
    )
    exp_setup: dict = exp_utils.get_exp_setup(exp_file_path, args)

    args["our_exp_folder"] = exp_setup["general"]["folder_name"]

    # Generate perturbation vectors
    generate_vectors(args, exp_setup)


if __name__ == "__main__":
    cfg.init_licence()
    # Get arguments
    mult_pert_arg_setup = a_parsers.MultiPerturbationArgSetup()
    mult_pert_arg_setup.setup_parser()
    args = mult_pert_arg_setup.args
    # g_ui.confirm_run_setup(args)

    main(args)
