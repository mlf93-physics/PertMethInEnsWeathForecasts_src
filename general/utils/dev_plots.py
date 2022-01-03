import matplotlib.pyplot as plt
import numpy as np
from general.params.model_licences import Models
import config as cfg

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    from shell_model_experiments.params.params import ParamsStructType
    from shell_model_experiments.params.params import PAR as PAR_SH

    params = PAR_SH
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params as l63_params

    params = l63_params


def dev_plot_perturbation_generation(perturb, perturb_temp):
    # Plot the random and the eigenvector scaled perturbation
    lambda_factor_temp = params.seeked_error_norm / np.linalg.norm(perturb_temp)
    perturb_temp = lambda_factor_temp * perturb_temp

    # Plot random perturbation
    # plt.plot(perturb_temp.real, 'b-')
    # plt.plot(perturb_temp.imag, 'r-')
    # Plot perturbation scaled along the eigenvector
    if cfg.MODEL == Models.SHELL_MODEL:
        plt.plot(perturb.real, "b--")
        plt.plot(perturb.imag, "r--")
        plt.legend(["Real part", "Imag part"])
        plt.xlabel("Shell number")
    elif cfg.MODEL == Models.LORENTZ63:
        plt.plot(perturb, "r--")
        plt.legend(["Perturbation"])
        plt.xlabel("Coordinate")

    plt.ylabel("Perturbation")
    plt.show()
