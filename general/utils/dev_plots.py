import matplotlib.pyplot as plt
import numpy as np
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params as l63_params
from general.params.model_licences import Models
from config import MODEL

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params


def dev_plot_perturbation_generation(perturb, perturb_temp):
    # Plot the random and the eigenvector scaled perturbation
    lambda_factor_temp = params.seeked_error_norm / np.linalg.norm(perturb_temp)
    perturb_temp = lambda_factor_temp * perturb_temp

    # Plot random perturbation
    # plt.plot(perturb_temp.real, 'b-')
    # plt.plot(perturb_temp.imag, 'r-')
    # Plot perturbation scaled along the eigenvector
    if MODEL == Models.SHELL_MODEL:
        plt.plot(perturb.real, "b--")
        plt.plot(perturb.imag, "r--")
        plt.legend(["Real part", "Imag part"])
        plt.xlabel("Shell number")
    elif MODEL == Models.LORENTZ63:
        plt.plot(perturb, "r--")
        plt.legend(["Perturbation"])
        plt.xlabel("Coordinate")

    plt.ylabel("Perturbation")
    plt.show()
