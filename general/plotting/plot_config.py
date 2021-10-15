import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker


def setup_plotting_defaults():
    print("Plotting with default settings")
    plt.rcParams["figure.figsize"] = [16, 9]


def latex_plot_settings():
    print("Plotting with latex settings")
    matplotlib.use("pgf")
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "figure.figsize": (4.7747, 3.5),
            "axes.titlesize": "medium",
            "axes.labelsize": "medium",
            "xtick.labelsize": "medium",
            "ytick.labelsize": "medium",
            "lines.linewidth": 1,
        }
    )
