import sys

sys.path.append("..")
import copy
import pathlib as pl

import config as cfg
import general.analyses.plot_analyses as g_plt_anal
import general.plotting.plot_config as g_plt_config
import general.utils.plot_utils as g_plt_utils
import general.plotting.plot_data as g_plt_data
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
import general.plotting.plot_config as plt_config
import general.utils.importing.import_perturbation_data as pt_import
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from general.params.model_licences import Models

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.plotting.plot_data as sh_plot
    import shell_model_experiments.utils.util_funcs as sh_utils
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType

    params = PAR_SH
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.plotting.plot_data as l63_plot

    params = l63_params


def tlm_spectrum(args):

    (
        u_stores,
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        header_dicts,
        u_ref_stores,
    ) = g_import.import_perturbation_velocities(args, search_pattern="*perturb*.csv")

    u_stores = np.array(u_stores)

    normed_u_stores = g_utils.normalize_array(u_stores, norm_value=1, axis=2)

    mean_normed_u_stores = np.mean(np.mean(normed_u_stores, axis=0), axis=0)

    plt.plot(np.log2(params.k_vec_temp), np.abs(mean_normed_u_stores))
    plt.yscale("log")


if __name__ == "__main__":
    cfg.init_licence()
    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    args = stand_plot_arg_parser.args

    g_ui.confirm_run_setup(args)

    # Shell model specific
    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        sh_utils.update_dependent_params(params)
        sh_utils.set_params(params, parameter="sdim", value=args["sdim"])
        sh_utils.update_arrays(params)

    plt_config.adjust_default_fig_axes_settings(args)

    if "tlm_spectrum" in args["plot_type"]:
        tlm_spectrum(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    g_plt_utils.save_or_show_plot(args)
