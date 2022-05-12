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
import scipy.ndimage as sp_ndi
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

    perturbed_fc = one_cycle_time_array ** 2

    coefficients = np.linspace(0.2, 1.0, n_cycles, endpoint=True)

    # Make axes
    axes = plt.axes()

    for i in range(n_cycles):
        adjusted_perturbed_fc = coefficients[i] * perturbed_fc + 0.25
        # Plot breed cycles
        axes.plot(
            one_cycle_time_array + i * cycle_length,
            adjusted_perturbed_fc,
            "k--",
        )
        # Plot vertical line indicators
        xpoint = one_cycle_time_array[-1] + i * cycle_length
        axes.plot(
            np.ones(2) * xpoint,
            [0, adjusted_perturbed_fc[-1]],
            "k",
            linestyle="dotted",
            linewidth=1,
        )
        axes.plot(
            np.ones(2) * i * cycle_length,
            [0, adjusted_perturbed_fc[0]],
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
    axes.set_ylabel("Error")
    # xticks = np.arange(-(n_cycles + 1), 1, dtype=np.int32)
    # axes.yaxis.set_major_locator(mticker.FixedLocator(xticks))
    axes.set_xticks(np.arange(0, 6))
    axes.set_xticklabels(
        ["$t_{-n}$", "$t_{-n + 1}$", "$t_{-n + 2}$", "$\\cdots$", "$t_{-1}$", "$t_0$"]
    )
    axes.set_xlabel("Time")

    if args["tolatex"]:
        plt_config.adjust_axes(axes)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath="thesis_figures/pt_methods",
            file_name="breeding_cycle_visualisation",
        )


def visualize_erosion_and_dilation(args):

    np.random.seed(20)
    n_entries = 20
    rand_array = np.round_(np.random.rand(n_entries), decimals=0).astype(np.int8)

    structure = np.ones(3)
    eroded_bool_array = sp_ndi.binary_erosion(
        rand_array, origin=0, structure=structure, border_value=0
    )
    dilated_bool_array = sp_ndi.binary_dilation(
        rand_array, origin=0, structure=structure, border_value=0
    )

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True)

    sb.heatmap(
        # np.arange(n_entries)[:, np.newaxis],
        rand_array[np.newaxis, :],
        ax=axes[0],
        annot=True,
        # annot_kws={"fmt": "%d"},
        cmap="Greys_r",
        cbar=False,
        linecolor="grey",
        linewidths=0.5,
    )

    # Drawing the frame
    sb.heatmap(
        # np.arange(n_entries)[:, np.newaxis],
        eroded_bool_array[np.newaxis, :],
        ax=axes[1],
        annot=True,
        # annot_kws={"fmt": "%d"},
        cmap="Greys_r",
        cbar=False,
        linecolor="grey",
        linewidths=0.5,
    )
    sb.heatmap(
        # np.arange(n_entries)[:, np.newaxis],
        dilated_bool_array[np.newaxis, :],
        ax=axes[2],
        annot=True,
        # annot_kws={"fmt": "%d"},
        cmap="Greys_r",
        cbar=False,
        linecolor="grey",
        linewidths=0.5,
    )

    # Set labels
    axes[0].set_ylabel("Original", rotation=0, horizontalalignment="right", y=0.3)
    axes[1].set_ylabel("Erosion", rotation=0, horizontalalignment="right", y=0.3)
    axes[2].set_ylabel("Dilation", rotation=0, horizontalalignment="right", y=0.3)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        draw_rectangle(n_entries, ax)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath="thesis_figures/appendices/region_analysis_shell_model/",
            file_name="erosion_dilation_visualization",
        )


def draw_rectangle(n_entries, axes):
    axes.axhline(y=0, xmin=4 / n_entries, xmax=7 / n_entries, color="r", linewidth=2)
    axes.axhline(y=1, xmin=4 / n_entries, xmax=7 / n_entries, color="r", linewidth=2)
    axes.axvline(x=4, color="r", linewidth=2)
    axes.axvline(x=7, color="r", linewidth=2)


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
    elif "erosion_and_dilation" in args["plot_type"]:
        visualize_erosion_and_dilation(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    g_plt_utils.save_or_show_plot(args)
