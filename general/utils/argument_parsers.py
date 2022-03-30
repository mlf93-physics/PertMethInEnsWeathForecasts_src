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

# Global variables
V_CHOICES = ["bv", "bv_eof", "sv", "fsv", "lv", "alv", "all"]
PT_CHOICES = ["bv", "bv_eof", "rd", "tl_rd", "nm", "sv", "rf", "lv", "all"]

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
        if not isinstance(self._args, dict):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        # Prepare model specific default arguments
        if cfg.MODEL == Models.SHELL_MODEL:
            # datapath = "./data/ny_n19/ny2.37e-08_ny_n19_t3.00e+02_n_f0_f1.0_kexp2"
            # datapath = "./data/new_dt/ny_n19/ny2.37e-08_ny_n19_t3.00e+02_n_f0_f1.0_sdim20_kexp2/"
            datapath = (
                "./data/thesis_data/ny2.37e-08_ny_n19_t3.00e+03_n_f0_f1.0_sdim20_kexp2/"
            )

        elif cfg.MODEL == Models.LORENTZ63:
            # datapath = "./data/sig1.00e+01_t9.10e+03_b2.67e+00_r2.80e+01/"
            datapath = (
                "./data/thesis_data/sig1.00e+01_t1.00e+05_b2.67e+00_r2.80e+01_dt0.01/"
            )

        # Add general arguments
        self._parser.add_argument("-dp", "--datapath", type=str, default=datapath)
        self._parser.add_argument(
            "--ref_data_out", type=str, default="./data/thesis_data"
        )
        self._parser.add_argument(
            "--analysis_path",
            type=str,
            default="./data/thesis_data/ny2.37e-08_ny_n19_t3.00e+03_n_f0_f1.0_sdim20_kexp2/analysis_data/",
        )
        self._parser.add_argument("-ttr", "--time_to_run", default=0.1, type=float)
        self._parser.add_argument("--burn_in_time", default=0.0, type=float)
        self._parser.add_argument("--seed_mode", action="store_true")
        self._parser.add_argument("--noconfirm", action="store_true")

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
            self._parser.add_argument(
                "--ny_n",
                default=19,
                type=int,
                help="The shell at which the diffusion sets in",
            )
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
        if not isinstance(self._args, dict):
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
        if not isinstance(self._args, dict):
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
        self._parser.add_argument(
            "--specific_runs_per_profile", nargs="+", default=None, type=int
        )
        self._parser.add_argument("--n_profiles", default=1, type=int)
        self._parser.add_argument("--exp_setup", default=None, type=str)
        start_time_group: argparse._MutuallyExclusiveGroup = (
            self._parser.add_mutually_exclusive_group()
        )
        start_time_group.add_argument(
            "--regime_start",
            type=str,
            choices=[None, "low", "high"],
            default=None,
            help="Whether to start in low or high predictability regime",
        )
        start_time_group.add_argument(
            "--start_times", nargs="+", type=float, default=None
        )
        self._parser.add_argument("--start_time_offset", default=None, type=float)

    def validate_arguments(self):
        if (
            self.args["regime_start"] is not None
            and cfg.MODEL is not Models.SHELL_MODEL
        ):
            self._parser.error(
                f"regime_start argument can only be used with the {str(Models.SHELL_MODEL)}"
            )


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
        if not isinstance(self._args, dict):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        # Add arguments
        self._parser.add_argument("--pert_vector_folder", default="", type=str)
        self._parser.add_argument("--specific_start_vector", default=0, type=int)
        self._parser.add_argument("--bv_raw_perts", action="store_true")


class PerturbationArgSetup:
    def __init__(self):
        self._parser = parser
        self._args = None

        # Setup standard setup
        __relref_arg_setup = RelReferenceArgSetup()
        __relref_arg_setup.setup_parser()
        __relref_arg_setup.validate_arguments()

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
        if not isinstance(self._args, dict):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        # Add optional arguments
        self._parser.add_argument("--endpoint", action="store_true")
        self._parser.add_argument("--save_no_pert", action="store_true")
        pert_mode_group = self._parser.add_mutually_exclusive_group(
            required=cfg.LICENCE
            in [EXP.NORMAL_PERTURBATION, EXP.HYPER_DIFFUSIVITY, EXP.BREEDING_VECTORS]
        )
        pert_mode_group.add_argument(
            "--pert_mode",
            choices=["rd", "nm", "lv", "bv", "bv_eof", "rf", "sv", "fsv"],
            type=str,
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
        # if (
        #     self.args["pert_mode"] in ["rd", "nm", "rf"]
        #     and self.args["start_times"] is None
        #     and self.args["regime_start"] is None
        # ):
        #     if cfg.LICENCE not in [EXP.VERIFICATION]:
        #         self._parser.error(
        #             "--start_times or --regime_start argument is required when pert_mode is 'rd' or 'nm'"
        #         )


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
        if not isinstance(self._args, dict):
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


class ComparisonArgParser:
    def __init__(self, setup_parents=True):
        self._parser = parser
        self._args = None

        if setup_parents:
            _temp_parser = MultiPerturbationArgSetup()
            _temp_parser.setup_parser()
            # Needed for RF perturbations to work
            _temp_parser = ReferenceAnalysisArgParser()
            _temp_parser.setup_parser()

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
        if not isinstance(self._args, dict):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        self._parser.add_argument(
            "-v",
            "--vectors",
            nargs="+",
            default=[],
            type=str,
            choices=V_CHOICES,
        )

        self._parser.add_argument(
            "-pt",
            "--perturbations",
            nargs="+",
            default=[],
            type=str,
            choices=PT_CHOICES,
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
        if not isinstance(self._args, dict):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        self._parser.add_argument("--ref_start_time", default=0, type=float)
        self._parser.add_argument("--ref_end_time", default=-1, type=float)
        self._parser.add_argument(
            "--specific_ref_records", nargs="+", default=[0], type=int
        )
        self._parser.add_argument("--n_ref_records", default=None, type=int)


class ComparisonPlottingArgParser:
    def __init__(self):
        self._parser = parser
        self._args = None

        _temp_parser = ComparisonArgParser(setup_parents=False)
        _temp_parser.setup_parser()

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
        if not isinstance(self._args, dict):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        self._parser.add_argument("--exp_folders", nargs="+", default=None, type=str)
        lv_compare_group: argparse._MutuallyExclusiveGroup = (
            self._parser.add_mutually_exclusive_group()
        )
        lv_compare_group.add_argument(
            "-nlvs",
            "--n_lvs_to_compare",
            type=int,
            help="Used to plot vector comparison to a specific number of LVs/ALVs",
        )
        lv_compare_group.add_argument(
            "-lvs",
            "--lvs_to_compare",
            nargs="+",
            type=int,
            help="Used to plot vector comparison to specific LVs/ALVs",
        )


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
        __relref_arg_setup.validate_arguments()

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
        if not isinstance(self._args, dict):
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
        self._parser.add_argument("-nt", "--notight", action="store_true")
        self._parser.add_argument("-s", "--save_fig", action="store_true")
        self._parser.add_argument(
            "--save_fig_name", type=str, help="Name of file to save figure to"
        )
        self._parser.add_argument(
            "--datapaths",
            nargs="+",
            type=str,
            default=None,
            help="For plots using multiple different datapaths (e.g. different reference files)",
        )
        self._parser.add_argument(
            "--tolatex",
            action="store_true",
            help="Used to prepare plot for latex report",
        )

        self._parser.add_argument(
            "-lf",
            "--latex_format",
            choices=[
                "normal_small",
                "normal_large",
                "horizontal_panel",
                "horizontal_panel_with_cbar",
                "two_panel",
                "three_panel",
                "quad_item",
                "two_vertical_panel",
                "two_quads",
                "full_page",
            ],
            type=str,
            help="Used to prepare plot for latex report",
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
        self._parser.add_argument(
            "--right_spine",
            action="store_true",
            help="Used to enable right spine",
        )

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

        sub_parser = self._parser.add_subparsers()
        plot_kwargs_parser: argparse.ArgumentParser = sub_parser.add_parser(
            "plot_kwargs"
        )
        plot_kwargs_parser.add_argument(
            "--display_type", type=str, help="Used to determine how a plot is displayed"
        )
        plot_kwargs_parser.add_argument(
            "--mark_pert_start",
            default=False,
            type=bool,
            help="Used to mark perturbation start times on e.g. energy plot",
        )
        plot_kwargs_parser.add_argument(
            "--ref_highlight",
            default=False,
            type=bool,
            help="L63: Used to make a highlight plot of a specific area of the attractor",
        )

        plot_kwargs_parser.add_argument(
            "--rmse_spread",
            action="store_true",
            help="Used to enable plotting spread together with RMSE instead of only RMSE",
        )
        plot_kwargs_parser.add_argument(
            "--elev",
            type=float,
            default=6,
            help="Elevation in 3D plot",
        )
        plot_kwargs_parser.add_argument(
            "--azim",
            type=float,
            default=-100,
            help="Azimuthal angle in 3D plot",
        )
        plot_kwargs_parser.add_argument(
            "--exp_growth_type",
            type=str,
            default="instant",
            choices=["instant", "mean"],
            help="The analysis type of the exponential growth comparison",
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
        if not isinstance(self._args, dict):
            self._args = vars(self._parser.parse_known_args()[0])

        return self._args

    def setup_parser(self):
        self._parser.add_argument(
            "--verification_type", default=None, required=True, type=str
        )
