"""Make plots of conceptual visualizations

Example
-------
python ../general/plotting/conceptual_visualizations.py
--plot_type=error_norm_compare

"""

import sys

sys.path.append("..")
import general.utils.argument_parsers as a_parsers
import general.utils.arg_utils as a_utils
import general.utils.plot_utils as g_plt_utils
from general.plotting.plot_params import *
import general.plotting.plot_config as plt_config
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sb
import numpy as np


def visualize_breeding_method(args):
    n_cycles = 5
    cycle_length = 1
    cycle_offset_ratio = 1 / 20
    n_points_per_cycle = 50
    one_cycle_time_array = np.linspace(
        0, cycle_length - cycle_offset_ratio * cycle_length, n_points_per_cycle
    )
    total_cycle_time_array = np.linspace(
        0, cycle_length * n_cycles, n_points_per_cycle * n_cycles
    )

    perturbed_fc = one_cycle_time_array ** 2 + 0.25

    # Make axes
    axes = plt.axes()

    for i in range(n_cycles):
        # Plot breed cycles
        axes.plot(one_cycle_time_array + i * cycle_length, perturbed_fc, "k--")
        # Plot vertical line indicators
        xpoint = one_cycle_time_array[-1] + i * cycle_length
        axes.plot(
            np.ones(2) * xpoint,
            [0, perturbed_fc[-1]],
            "k",
            linestyle="dotted",
            linewidth=1,
        )
        axes.plot(
            np.ones(2) * i * cycle_length,
            [0, perturbed_fc[0]],
            "k",
            linestyle="dotted",
            linewidth=1,
        )

    # Plot "reference"
    axes.plot(total_cycle_time_array, np.zeros(n_points_per_cycle * n_cycles), "k-")
    axes.plot(np.arange(n_cycles), np.zeros(n_cycles), "kx", markersize=4)
    axes.set_ylim(-0.25, 1.5)
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.set_yticklabels([])
    axes.set_yticks([])
    xticks = np.arange(-(n_cycles + 1), 1, dtype=np.int32)
    # axes.yaxis.set_major_locator(mticker.FixedLocator(xticks))
    axes.set_xticklabels(xticks)
    axes.set_xlabel("Time")

    if args["tolatex"]:
        plt_config.adjust_axes(axes)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath="thesis_figures/pt_methods",
            file_name="breeding_cycle_visualisation",
        )


if __name__ == "__main__":

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    compare_plot_arg_parser = a_parsers.ComparisonPlottingArgParser()
    compare_plot_arg_parser.setup_parser()
    args: dict = compare_plot_arg_parser.args

    a_utils.react_on_comparison_arguments(args)

    plt_config.adjust_default_fig_axes_settings(args)

    if "breed_method" in args["plot_type"]:
        visualize_breeding_method(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    g_plt_utils.save_or_show_plot(args)
