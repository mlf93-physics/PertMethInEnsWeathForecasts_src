import numpy as np
import matplotlib.pyplot as plt
from shell_model_experiments.params.params import ParamsStructType
from shell_model_experiments.params.params import PAR


def dev_plot_eigen_mode_analysis(
    e_values, J_matrix, e_vectors, header=None, perturb_pos=None
):

    if header is not None:
        title_append = (
            f"; $\\nu$={header['ny']:.2e}, f={header['f']}, "
            + f"position={perturb_pos/PAR.sample_rate*PAR.dt:.2f}s"
        )
    else:
        title_append = ""

    sort_id = e_values.argsort()[::-1]

    # Plot eigenvalues
    plt.figure()
    plt.scatter(e_values.real, e_values.imag, color="b", marker="x")
    plt.scatter(e_values.real, e_values.conj().imag, color="r", marker="x")
    plt.xlabel("Real part")
    plt.ylabel("Imaginary part")
    plt.title("The eigenvalues" + title_append)
    plt.grid()

    # Plot J_matrix
    plt.figure()
    plt.pcolormesh(np.log(np.abs(J_matrix)), cmap="Reds")
    # plt.ylim(20, 0)
    plt.clim(0, None)
    plt.xlabel("Shell number; $n$")
    plt.ylabel("Shell number; $m$")
    plt.title("Mod of the components of the Jacobian" + title_append)
    plt.colorbar()

    reprod_J_matrix = e_vectors @ np.diag(e_values) @ np.linalg.inv(e_vectors)

    plt.figure()
    plt.pcolormesh(np.abs(reprod_J_matrix), cmap="Reds")
    # plt.ylim(20, 0)
    plt.clim(0, None)
    plt.xlabel("Shell number; $n$")
    plt.ylabel("Shell number; $m$")
    plt.title("Reproduced J_matrix" + title_append)
    plt.colorbar()

    # Plot eigenvectors
    plt.figure()
    plt.pcolormesh(np.abs(e_vectors[:, sort_id]) ** 2, cmap="Reds")
    plt.xlabel("Lyaponov index; $j$")
    plt.ylabel("Shell number; $i$")
    plt.title("Mod squared of the components of the eigenvectors" + title_append)
    plt.colorbar()

    # plt.figure()
    # plt.plot(e_values[sort_id].real, 'k.')
    # plt.plot(e_values[sort_id].imag, 'r.')
    # # plt.xlabel('Lyaponov index; $j$')
    # # plt.ylabel('Shell number; $i$')
    # # plt.title('Mod squared of the components of the eigenvectors' + title_append)
    # # plt.colorbar()
    plt.show()
