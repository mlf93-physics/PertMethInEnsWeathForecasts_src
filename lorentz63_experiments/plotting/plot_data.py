from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt


def plot_attractor(data):
    # Setup axes
    ax = plt.axes(projection="3d")

    ax.plot3D(data[:, 1], data[:, 2], data[:, 3], "k-", linewidth=0.5)
    plt.show()
