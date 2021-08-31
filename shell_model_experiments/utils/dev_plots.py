import numpy as np
import matplotlib.pyplot as plt
from src.params.params import *

def dev_plot_eigen_mode_analysis(e_values, J_matrix, e_vectors, header=None,
        perturb_pos=None):

    if header is not None:
        title_append = f"; $\\nu$={header['ny']:.2e}, f={header['f']}, "+\
            f'position={perturb_pos/sample_rate*dt:.2f}s'
    else:
        title_append = ""

    sort_id = e_values.argsort()[::-1]

    # Plot eigenvalues
    plt.figure()
    plt.scatter(e_values.real, e_values.imag, color='b', marker='x')
    plt.scatter(e_values.real, e_values.conj().imag, color='r', marker='x')
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.title('The eigenvalues' + title_append)
    plt.grid()

    # Plot J_matrix
    plt.figure()
    plt.pcolormesh(np.log(np.abs(J_matrix)), cmap='Reds')
    # plt.ylim(20, 0)
    plt.clim(0, None)
    plt.xlabel('Shell number; $n$')
    plt.ylabel('Shell number; $m$')
    plt.title('Mod of the components of the Jacobian' + title_append)
    plt.colorbar()

    reprod_J_matrix = e_vectors @ np.diag(e_values) @ np.linalg.inv(e_vectors)

    plt.figure()
    plt.pcolormesh(np.abs(reprod_J_matrix), cmap='Reds')
    # plt.ylim(20, 0)
    plt.clim(0, None)
    plt.xlabel('Shell number; $n$')
    plt.ylabel('Shell number; $m$')
    plt.title('Reproduced J_matrix' + title_append)
    plt.colorbar()


    # Plot eigenvectors
    plt.figure()
    plt.pcolormesh(np.abs(e_vectors[:, sort_id])**2, cmap='Reds')
    plt.xlabel('Lyaponov index; $j$')
    plt.ylabel('Shell number; $i$')
    plt.title('Mod squared of the components of the eigenvectors' + title_append)
    plt.colorbar()

    

    # plt.figure()
    # plt.plot(e_values[sort_id].real, 'k.')
    # plt.plot(e_values[sort_id].imag, 'r.')
    # # plt.xlabel('Lyaponov index; $j$')
    # # plt.ylabel('Shell number; $i$')
    # # plt.title('Mod squared of the components of the eigenvectors' + title_append)
    # # plt.colorbar()
    plt.show()


def dev_plot_perturbation_generation(perturb, perturb_temp):
    # Plot the random and the eigenvector scaled perturbation
    lambda_factor_temp = seeked_error_norm/np.linalg.norm(perturb_temp)
    perturb_temp = lambda_factor_temp*perturb_temp
    
    # Plot random perturbation
    # plt.plot(perturb_temp.real, 'b-')
    # plt.plot(perturb_temp.imag, 'r-')
    # Plot perturbation scaled along the eigenvector
    plt.plot(perturb.real, 'b--')
    plt.plot(perturb.imag, 'r--')
    plt.legend(['Real part', 'Imag part'])
    plt.xlabel('Shell number')
    plt.ylabel('Perturbation')
    plt.show()