__all__ = [
    "StandardArgSetup",
    "StandardModelArgSetup",
    "StandardRunnerArgSetup",
    "RelReferenceArgSetup",
    "PerturbationVectorArgSetup",
    "PerturbationArgSetup",
    "MultiPerturbationArgSetup",
    "ReferenceAnalysisArgParser",
    "ComparisonPlottingArgParser",
    "StandardPlottingArgParser",
]

import sys

sys.path.append("..")
import argparse
import numpy as np
from general.params.experiment_licences import Experiments as EXP
from general.params.model_licences import Models
import config as cfg

# Instantiate ArgumentParser
parser = argparse.ArgumentParser()


class StandardArgSetup:
    def __init__(self):
        self._parser = parser
        self._args = None

    @property
    def args(self):
        """The vars(parsed arguments.)

        The parsed args are saved to a local attribute if not already present
        to avoid multiple calls to parse_args()

        Returns
        -------
        argparse.Namespace
            The parsed arguments
        """
        if not isinstance(self._args, argparse.Namespace):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        # Prepare model specific default arguments
        if cfg.MODEL == Models.SHELL_MODEL:
            datapath = "./data/ny_n19/ny2.37e-08_ny_n19_t3.00e+02_n_f0_f1.0_kexp2"

        elif cfg.MODEL == Models.LORENTZ63:
            datapath = "./data/sig1.00e+01_t9.10e+03_b2.67e+00_r2.80e+01/"

        # Add general arguments
        self._parser.add_argument("-dp", "--datapath", type=str, default=datapath)
        self._parser.add_argument("-ttr", "--time_to_run", default=0.1, type=float)
        self._parser.add_argument("--burn_in_time", default=0.0, type=float)
        self._parser.add_argument("--seed_mode", action="store_true")

    def react_on_arguments(self):
        # Set seed if wished
        if self.args["seed_mode"]:
            np.random.seed(seed=1)


class StandardModelArgSetup:
    def __init__(self):
        self._parser = parser
        self._args = None

    def setup_parser(self):
        # Add model specific arguments
        if cfg.MODEL == Models.SHELL_MODEL:
            self._parser.add_argument("--ny_n", default=19, type=int)
            self._parser.add_argument("--forcing", default=1, type=float)
            self._parser.add_argument("--diff_exponent", default=2, type=int)
            self._parser.add_argument(
                "--diff_type",
                choices=["normal", "inf_hyper"],
                default="normal",
                type=str,
            )
            self._parser.add_argument("--sdim", default=20, type=int)

        elif cfg.MODEL == Models.LORENTZ63:
            self._parser.add_argument("--sigma", default=10, type=float)
            self._parser.add_argument("--r_const", default=28.0, type=float)
            self._parser.add_argument("--b_const", default=8 / 3, type=float)


class StandardRunnerArgSetup:
    def __init__(self):
        self._parser = parser
        self._args = None

        __standard_arg_setup = StandardArgSetup()
        __standard_arg_setup.setup_parser()
        __standard_arg_setup.react_on_arguments()
        __standard_model_arg_setup = StandardModelArgSetup()
        __standard_model_arg_setup.setup_parser()

    @property
    def args(self):
        """The vars(parsed arguments.)

        The parsed args are saved to a local attribute if not already present
        to avoid multiple calls to parse_args()

        Returns
        -------
        argparse.Namespace
            The parsed arguments
        """
        if not isinstance(self._args, argparse.Namespace):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        # Add general arguments
        self._parser.add_argument("--erda_run", action="store_true")
        self._parser.add_argument("--skip_save_data", action="store_true")
        self._parser.add_argument("--noprint", action="store_true")


class RelReferenceArgSetup:
    def __init__(self, setup_parents=True):
        self._parser = parser
        self._args = None

        # Setup standard setup
        if setup_parents:
            __standard_runner_arg_setup = StandardRunnerArgSetup()
            __standard_runner_arg_setup.setup_parser()

    @property
    def args(self):
        """The vars(parsed arguments.)

        The parsed args are saved to a local attribute if not already present
        to avoid multiple calls to parse_args()

        Returns
        -------
        argparse.Namespace
            The parsed arguments
        """
        if not isinstance(self._args, argparse.Namespace):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        # Add required arguments
        self._parser.add_argument("--exp_folder", required=False, type=str)
        self._parser.add_argument(
            "--out_exp_folder", required=False, default="temp_exp_folder", type=str
        )
        # Add optional arguments
        self._parser.add_argument("--n_runs_per_profile", default=1, type=int)
        self._parser.add_argument("--n_profiles", default=1, type=int)
        self._parser.add_argument("--start_times", nargs="+", type=float, default=None)
        self._parser.add_argument("--start_time_offset", default=None, type=float)
        self._parser.add_argument("--exp_setup", default=None, type=str)


class PerturbationVectorArgSetup:
    def __init__(self):
        self._parser = parser
        self._args = None

    @property
    def args(self):
        """The vars(parsed arguments.)

        The parsed args are saved to a local attribute if not already present
        to avoid multiple calls to parse_args()

        Returns
        -------
        argparse.Namespace
            The parsed arguments
        """
        if not isinstance(self._args, argparse.Namespace):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        # Add arguments
        self._parser.add_argument("--pert_vector_folder", default=None, type=str)
        self._parser.add_argument("--specific_start_vector", default=0, type=int)


class PerturbationArgSetup:
    def __init__(self):
        self._parser = parser
        self._args = None

        # Setup standard setup
        __relref_arg_setup = RelReferenceArgSetup()
        __relref_arg_setup.setup_parser()

        # Setup perturbation vector args
        __pert_vector_arg_setup = PerturbationVectorArgSetup()
        __pert_vector_arg_setup.setup_parser()

    @property
    def args(self):
        """The vars(parsed arguments.)

        The parsed args are saved to a local attribute if not already present
        to avoid multiple calls to parse_args()

        Returns
        -------
        argparse.Namespace
            The parsed arguments
        """
        if not isinstance(self._args, argparse.Namespace):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        # Add optional arguments
        self._parser.add_argument("--endpoint", action="store_true")
        pert_mode_group = self._parser.add_mutually_exclusive_group(
            required=cfg.LICENCE
            in [EXP.NORMAL_PERTURBATION, EXP.HYPER_DIFFUSIVITY, EXP.BREEDING_VECTORS]
        )
        pert_mode_group.add_argument(
            "--pert_mode", choices=["rd", "nm", "bv", "bv_eof", "sv"], type=str
        )

        # Add model specific arguments
        if cfg.MODEL == Models.SHELL_MODEL:
            pert_mode_group.add_argument("--single_shell_perturb", type=int)

    def validate_arguments(self):
        # Test if start_time is set when start_tim_offset is set
        if (
            self.args["start_time_offset"] is not None
            and self.args["start_times"] is None
        ):
            self._parser.error(
                "--start_times argument is required when --start_time_offset is set"
            )

        if self.args["pert_mode"] in ["bv", "bv_eof"]:
            # Test if pert_vector_folder is set if pert_mode is in ["bv", "bv_eof"]
            if self.args["pert_vector_folder"] is None:
                self._parser.error(
                    "--pert_vector_folder argument is required when"
                    + " --pert_mode is one of ['bv', 'bv_eof]"
                )

        # Test if start_times is set when pert_mode in ["rd", "nm"]
        if self.args["pert_mode"] in ["rd", "nm"] and self.args["start_times"] is None:
            self._parser.error(
                "--start_times argument is required when pert_mode is 'rd' or 'nm'"
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
        """The vars(parsed arguments.)

        The parsed args are saved to a local attribute if not already present
        to avoid multiple calls to parse_args()

        Returns
        -------
        argparse.Namespace
            The parsed arguments
        """
        if not isinstance(self._args, argparse.Namespace):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        # Add optional arguments
        self._parser.add_argument(
            "--save_last_pert",
            action="store_true",
            help="Save only the data for the last perturbation of a unit",
        )
        n_units_group = self._parser.add_mutually_exclusive_group()
        n_units_group.add_argument("--n_units", default=np.inf, type=int)
        n_units_group.add_argument(
            "--specific_units", nargs="+", default=None, type=int
        )


class ReferenceAnalysisArgParser:
    def __init__(self):
        self._parser = parser
        self._args = None

    @property
    def args(self):
        """The vars(parsed arguments.)

        The parsed args are saved to a local attribute if not already present
        to avoid multiple calls to parse_args()

        Returns
        -------
        argparse.Namespace
            The parsed arguments
        """
        if not isinstance(self._args, argparse.Namespace):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        self._parser.add_argument("--ref_start_time", default=0, type=float)
        self._parser.add_argument("--ref_end_time", default=-1, type=float)
        self._parser.add_argument(
            "--specific_ref_records", nargs="+", default=[0], type=int
        )


class ComparisonPlottingArgParser:
    def __init__(self):
        self._parser = parser
        self._args = None

    @property
    def args(self):
        """The vars(parsed arguments.)

        The parsed args are saved to a local attribute if not already present
        to avoid multiple calls to parse_args()

        Returns
        -------
        argparse.Namespace
            The parsed arguments
        """
        if not isinstance(self._args, argparse.Namespace):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        self._parser.add_argument("--exp_folders", nargs="+", default=None, type=str)


class StandardPlottingArgParser:
    def __init__(self):
        self._parser = parser
        self._args = None

        # Setup standard setup
        __standard_arg_setup = StandardArgSetup()
        __standard_arg_setup.setup_parser()
        __standard_arg_setup.react_on_arguments()

        __standard_model_arg_setup = StandardModelArgSetup()
        __standard_model_arg_setup.setup_parser()

        # Add arguments from MultiPerturbationArgSetup
        __multi_pert_arg_setup = MultiPerturbationArgSetup(setup_parents=False)
        __multi_pert_arg_setup.setup_parser()

        # Setup perturbation vector args
        __pert_vector_arg_setup = PerturbationVectorArgSetup()
        __pert_vector_arg_setup.setup_parser()

        # Add arguments for reference analysis
        __ref_arg_setup = ReferenceAnalysisArgParser()
        __ref_arg_setup.setup_parser()

        __relref_arg_setup = RelReferenceArgSetup(setup_parents=False)
        __relref_arg_setup.setup_parser()

    @property
    def args(self):
        """The vars(parsed arguments.)

        The parsed args are saved to a local attribute if not already present
        to avoid multiple calls to parse_args()

        Returns
        -------
        argparse.Namespace
            The parsed arguments
        """
        if not isinstance(self._args, argparse.Namespace):
            self._args = vars(self._parser.parse_known_args()[0])

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
        self._parser.add_argument("-s", "--save_fig", action="store_true")
        self._parser.add_argument(
            "--datapaths",
            nargs="+",
            type=str,
            default=None,
            help="For plots using multiple different datapaths (e.g. different reference files)",
        )

        # x, y limits
        self._parser.add_argument("--xlim", nargs=2, default=None, type=float)
        self._parser.add_argument("--ylim", nargs=2, default=None, type=float)
        self._parser.add_argument("--sharey", action="store_true")
        self._parser.add_argument("--average", action="store_true")

        # experiment plot speicific
        self._parser.add_argument("--n_files", default=np.inf, type=int)
        self._parser.add_argument("--file_offset", default=0, type=int)
        self._parser.add_argument("--specific_files", nargs="+", default=None, type=int)
        self._parser.add_argument("--combinations", action="store_true")
        self._parser.add_argument("--endpoint", action="store_true")

        if cfg.MODEL == Models.SHELL_MODEL:
            self._parser.add_argument(
                "--shell_cutoff",
                default=None,
                type=int,
                help="Defines which shells (all shells below shell_cutoff)"
                + " shall be compared to reference shells. Used in hyper diffusion"
                + " experiments, where only region of shells below diffusion region"
                + " is relevant",
            )


class VerificationArgParser:
    def __init__(self):
        self._parser = parser
        self._args = None

    @property
    def args(self):
        """The vars(parsed arguments.)

        The parsed args are saved to a local attribute if not already present
        to avoid multiple calls to parse_args()

        Returns
        -------
        argparse.Namespace
            The parsed arguments
        """
        if not isinstance(self._args, argparse.Namespace):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        self._parser.add_argument(
            "--verification_type", default=None, required=True, type=str
        )
