"""Make plots related to the RF perturbations

Example
-------


"""

import sys

sys.path.append("..")
import pathlib as pl

import config as cfg
import general.analyses.breed_vector_eof_analysis as bv_analysis
import general.analyses.plot_analyses as g_plt_anal
import general.plotting.plot_data as g_plt_data
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.perturb_utils as pt_utils
import general.utils.importing.import_utils as g_imp_utils
import general.utils.plot_utils as g_plt_utils
import general.utils.user_interface as g_ui
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
from general.params.model_licences import Models
from general.utils.module_import.type_import import *
from mpl_toolkits import mplot3d
from pyinstrument import Profiler

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.plotting.plot_data as sh_plot
    import shell_model_experiments.utils.util_funcs as sh_utils
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType

    params = PAR_SH
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
    import lorentz63_experiments.plotting.plot_data as l63_plot

    params = l63_params


def plot_rf_perturbation_vectors(args: dict, axes: plt.Axes = None):
    rand_field_perturbations = pt_utils.get_rand_field_perturbations(args)

    rand_field_perturbations_real = np.mean(
        np.abs(rand_field_perturbations.real), axis=1
    )
    rand_field_perturbations_imag = np.mean(
        np.abs(rand_field_perturbations.imag), axis=1
    )

    # Prepare axes
    if axes is None:
        axes = plt.axes()

    real_part_lines = axes.plot(
        np.log2(params.k_vec_temp),
        rand_field_perturbations_real,
        color="b",
        linestyle="solid",
        alpha=0.6,
    )
    imag_part_lines = axes.plot(
        np.log2(params.k_vec_temp),
        rand_field_perturbations_imag,
        color="r",
        linestyle="solid",
        alpha=0.6,
    )
    real_part_lines[0].set_label("Real part")
    imag_part_lines[0].set_label("Imag part")
    axes.xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))

    axes.set_xlabel("Shell index")
    axes.set_ylabel("Components")
    axes.set_yscale("log")
    axes.legend()
    axes.set_title(f"RF perturbations | $n_{{profiles}}$={args['n_profiles']}")


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    args = stand_plot_arg_parser.args

    g_ui.confirm_run_setup(args)

    # Make profiler
    profiler = Profiler()
    profiler.start()

    # Shell model specific
    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        sh_utils.update_dependent_params(params)
        sh_utils.set_params(params, parameter="sdim", value=args["sdim"])
        sh_utils.update_arrays(params)

    if "perturbations":
        plot_rf_perturbation_vectors(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    profiler.stop()
    print(profiler.output_text(color=True))

    g_plt_utils.save_or_show_plot(args)
