__all__ = [
    "StandardArgSetup",
    "StandardRunnerArgSetup",
    "PerturbationArgSetup",
    "MultiPerturbationArgSetup",
]

import sys

sys.path.append("..")
import argparse
import numpy as np
from general.params.model_licences import Models
from general.params.experiment_licences import Experiments as EXP
from config import MODEL, LICENCE

# Instantiate ArgumentParser
parser = argparse.ArgumentParser()


class StandardArgSetup:
    def __init__(self):
        self._parser = parser

    @property
    def args(self):
        return self._parser.parse_args()

    def setup_parser(self):
        # Prepare model specific default arguments
        if MODEL == Models.SHELL_MODEL:
            datapath = "./data/ny2.37e-08_t4.00e+02_n_f0_f1.0/"

        elif MODEL == Models.LORENTZ63:
            datapath = "./data/sig1.00e+01_t9.10e+03_b2.67e+00_r2.80e+01/"

        # Add general arguments
        self._parser.add_argument("-dp", "--path", type=str, default=datapath)
        self._parser.add_argument("-ttr", "--time_to_run", default=0.1, type=float)
        self._parser.add_argument("--burn_in_time", default=0.0, type=float)
        self._parser.add_argument("--seed_mode", action="store_true")


class StandardRunnerArgSetup:
    def __init__(self):
        self._parser = parser

        __standard_arg_setup = StandardArgSetup
        __standard_arg_setup.setup_parser()

    @property
    def args(self):
        return self._parser.parse_args()

    def setup_parser(self):
        # Add general arguments
        self._parser.add_argument("--erda_run", action="store_true")
        self._parser.add_argument("--skip_save_data", action="store_true")

        # Add model specific arguments
        if MODEL == Models.SHELL_MODEL:
            self._parser.add_argument("--ny_n", default=19, type=int)
            self._parser.add_argument("--forcing", default=1, type=float)

        elif MODEL == Models.LORENTZ63:
            self._parser.add_argument("--sigma", default=10, type=float)
            self._parser.add_argument("--r_const", default=28, type=float)
            self._parser.add_argument("--b_const", default=8 / 3, type=float)


class PerturbationArgSetup:
    def __init__(self):
        self._parser = parser
        self._args = None

        # Setup standard setup
        stand_arg_setup = StandardRunnerArgSetup()
        stand_arg_setup.setup_parser()

    @property
    def args(self):
        """The parsed arguments.

        The parsed args are saved to a local attribute if not already present
        to avoid multiple calls to parse_args()

        Returns
        -------
        argparse.Namespace
            The parsed arguments
        """
        if not isinstance(self._args, argparse.Namespace):
            self._args = self._parser.parse_known_args()[0]

        return self._args

    def setup_parser(self):
        # Add required arguments
        self._parser.add_argument("--exp_folder", required=False, type=str)
        # Add optional arguments
        self._parser.add_argument("--n_runs_per_profile", default=1, type=int)
        self._parser.add_argument("--n_profiles", default=1, type=int)
        self._parser.add_argument(
            "--pert_mode", default="random", choices=["nm", "bv"], type=str
        )
        self._parser.add_argument("--vectors", default=None, type=str)
        self._parser.add_argument(
            "--start_time",
            nargs="+",
            type=float,
            required=LICENCE == EXP.NORMAL_PERTURBATION,
        )
        self._parser.add_argument("--start_time_offset", default=None, type=float)
        self._parser.add_argument("--endpoint", action="store_true")
        self._parser.add_argument("--exp_setup", default=None, type=str)

        # Add model specific arguments
        if MODEL == Models.SHELL_MODEL:
            self._parser.add_argument("--single_shell_perturb", default=None, type=int)

    def validate_arguments(self):
        if self.args.start_time_offset is not None and self.args.start_times is None:
            self._parser.error(
                "--start_times argument is required when --start_time_offset is set"
            )

        if self.args.pert_mode == "bv" and self.args.pert_vectors is None:
            self._parser.error(
                "--pert_vectors argument is required when"
                + " --pert_mode is one of ['bv']"
            )


class MultiPerturbationArgSetup:
    def __init__(self, setup_parents=True):
        self._parser = parser
        self._args = None

        # Setup parent perturbation setup
        if setup_parents:
            __pert_arg_setup = PerturbationArgSetup()
            __pert_arg_setup.setup_parser()
            __pert_arg_setup.validate_arguments()

    @property
    def args(self):
        """The parsed arguments.

        The parsed args are saved to a local attribute if not already present
        to avoid multiple calls to parse_args()

        Returns
        -------
        argparse.Namespace
            The parsed arguments
        """
        if not isinstance(self._args, argparse.Namespace):
            self._args = self._parser.parse_known_args()[0]

        return self._args

    def setup_parser(self):
        # Add optional arguments
        num_units_group = self._parser.add_mutually_exclusive_group()
        num_units_group.add_argument("--num_units", default=np.inf, type=int)
        num_units_group.add_argument(
            "--specific_units", nargs="+", default=None, type=int
        )


class StandardPlottingArgParser:
    def __init__(self):
        self._parser = parser
        self._args = None

        # Setup standard setup
        __standard_arg_setup = StandardArgSetup()
        __standard_arg_setup.setup_parser()
        # Add arguments from MultiPerturbationArgSetup
        __multi_pert_arg_setup = MultiPerturbationArgSetup(setup_parents=False)
        __multi_pert_arg_setup.setup_parser()

    @property
    def args(self):
        """The parsed arguments.

        The parsed args are saved to a local attribute if not already present
        to avoid multiple calls to parse_args()

        Returns
        -------
        argparse.Namespace
            The parsed arguments
        """
        if not isinstance(self._args, argparse.Namespace):
            self._args = self._parser.parse_known_args()[0]

        return self._args

    def setup_parser(self):

        # Add optional arguments
        # General plot args
        self._parser.add_argument("--plot_type", nargs="+", default=None, type=str)
        self._parser.add_argument(
            "--plot_mode",
            nargs="?",
            default="standard",
            type=str,
            help="standard : plot everything with the standard plot setup;"
            + " detailed : plot extra details in plots",
        )
        self._parser.add_argument("-np", "--noplot", action="store_true")

        # x, y and time limits
        self._parser.add_argument("--ref_start_time", default=0, type=float)
        self._parser.add_argument("--ref_end_time", default=-1, type=float)
        self._parser.add_argument("--xlim", nargs=2, default=None, type=float)
        self._parser.add_argument("--ylim", nargs=2, default=None, type=float)
        self._parser.add_argument(
            "--specific_ref_records", nargs="+", default=[0], type=int
        )
        self._parser.add_argument("--sharey", action="store_true")

        # If running perturbations before plotting
        self._parser.add_argument("--start_time", nargs="+", type=float)
        self._parser.add_argument("--n_profiles", default=1, type=int)
        self._parser.add_argument("--n_runs_per_profile", default=1, type=int)

        # experiment plot speicific
        self._parser.add_argument("--exp_folder", nargs="?", default=None, type=str)
        self._parser.add_argument("--n_files", default=np.inf, type=int)
        self._parser.add_argument("--file_offset", default=0, type=int)
        self._parser.add_argument("--specific_files", nargs="+", default=None, type=int)
        self._parser.add_argument("--combinations", action="store_true")
        self._parser.add_argument("--endpoint", action="store_true")
