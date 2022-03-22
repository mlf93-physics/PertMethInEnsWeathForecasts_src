import numpy as np
from general.utils.module_import.type_import import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker

plt.rcParams["savefig.dpi"] = 300


def two_panel_figure():
    plt.rcParams["figure.figsize"] = [5.39749, 3]


def three_panel_figure():
    plt.rcParams["figure.figsize"] = [5.39749, 5.3]


def normal_figure():
    plt.rcParams["figure.figsize"] = [5.39749, 4.5]


def two_quads():
    plt.rcParams["figure.figsize"] = [5.39749 / 2, 5.39749 / 2]


def horizontal_panel_figure():
    plt.rcParams["figure.figsize"] = [5.39749, 1.5]


def horizontal_panel_with_cbar_figure():
    plt.rcParams["figure.figsize"] = [5.39749, 2.3]


def default_figure_settings(args):
    matplotlib.rcParams.update(
        {
            "axes.titlesize": "medium",
            "axes.labelsize": "medium",
            "xtick.labelsize": "small",
            "axes.spines.right": args["right_spine"],
            "axes.spines.top": False,
            "font.family": "serif",
            "text.usetex": True,
        }
    )


def adjust_axes(axes: Union[plt.Axes, List[plt.Axes]]):
    if isinstance(axes, list) or isinstance(axes, np.ndarray):
        for ax in axes:
            ax.set_title("")
    else:
        axes.set_title("")


def hide_axis_labels(axes):
    if isinstance(axes, list) or isinstance(axes, np.ndarray):
        for ax in axes:
            ax.set_xlabel("")
            ax.set_ylabel("")
    else:
        axes.set_xlabel("")
        axes.set_ylabel("")


def hide_axis_ticks(axes):
    if isinstance(axes, list) or isinstance(axes, np.ndarray):
        for ax in axes:
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
    else:
        axes.get_xaxis().set_ticks([])
        axes.get_yaxis().set_ticks([])


def adjust_axes_to_subplot(axes):
    axes.set_xlabel("")
    axes.set_xticklabels([])


def set_ytick_format(axes, fmt="%.0f"):
    axes.yaxis.set_major_formatter(mpl_ticker.FormatStrFormatter(fmt))


def adjust_default_fig_axes_settings(args):
    if args["tolatex"]:
        default_figure_settings(args)
        if args["latex_format"] == "horizontal_panel":
            horizontal_panel_figure()
        if args["latex_format"] == "horizontal_panel_with_cbar":
            horizontal_panel_with_cbar_figure()
        if args["latex_format"] == "two_panel":
            two_panel_figure()
        if args["latex_format"] == "three_panel":
            three_panel_figure()
        if args["latex_format"] == "normal":
            normal_figure()
        if args["latex_format"] == "two_quads":
            two_quads()
