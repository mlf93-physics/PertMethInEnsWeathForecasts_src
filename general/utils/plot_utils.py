import matplotlib.pyplot as plt
import numpy as np


def get_non_repeating_colors(n_colors=1):
    colormap = plt.cm.gist_rainbow
    cmap_list = [colormap(i) for i in np.linspace(0, 1, n_colors)]

    return cmap_list
