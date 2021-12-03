import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import seaborn as sb
from pyinstrument import Profiler
import numpy as np
import general.utils.perturb_utils as pt_utils
import general.utils.util_funcs as g_utils
import shell_model_experiments.utils.util_funcs as sh_utils
import general.utils.argument_parsers as a_parsers
from general.params.model_licences import Models
import config as cfg

import matplotlib.pyplot as plt

if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.utils.special_params as sh_sparams
    from shell_model_experiments.params.params import PAR
    from shell_model_experiments.params.params import ParamsStructType

    # Get parameters for model
    params = PAR
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams

    # Get parameters for model
    params = l63_params
    sparams = l63_sparams

profiler = Profiler()


def verify_lanczos_algorithm():
    # Define parameters
    # params.sdim = 10
    # params.seeked_error_norm = 1
    n_vectors = params.sdim
    n_runs = 100
    # Define base matrix
    base_matrix = np.eye(params.sdim)
    base_matrix = (
        base_matrix * np.arange(1, params.sdim + 1, dtype=sparams.dtype)[:, np.newaxis]
    )
    symmetric_matrix = base_matrix.T.conj() @ base_matrix

    # Numpy's eigenvalues
    np_e_values, np_e_vectors = np.linalg.eig(symmetric_matrix)
    sort_index = np.argsort(np_e_values)[::-1]
    np_e_values = np_e_values[sort_index]
    np_e_vectors = np_e_vectors[sort_index]

    # Prepare storage of sv vectors and -values
    sv_matrix_store = np.zeros((n_runs, params.sdim, n_vectors), dtype=sparams.dtype)
    s_values_store = np.zeros((n_runs, n_vectors), dtype=np.complex128)

    for i in range(n_runs):
        # Initiate the Lanczos arrays and algorithm
        propagated_vector: np.ndarray((params.sdim, 1)) = np.zeros(
            (params.sdim, 1), dtype=sparams.dtype
        )
        input_vector: np.ndarray((params.sdim, 1)) = np.zeros(
            (params.sdim, 1), dtype=sparams.dtype
        )
        lanczos_iterator = pt_utils.lanczos_vector_algorithm(
            propagated_vector=propagated_vector,
            input_vector_j=input_vector,
            n_iterations=n_vectors,
        )
        out_vector = np.random.rand(params.sdim).astype(sparams.dtype)
        out_vector = np.reshape(
            g_utils.normalize_array(out_vector, norm_value=1), (params.sdim, 1)
        )

        for _ in range(n_vectors):
            # Update arrays for the lanczos algorithm
            propagated_vector[:, :] = symmetric_matrix @ out_vector
            input_vector[:, :] = out_vector
            out_vector, tridiag_matrix, input_vector_matrix = next(lanczos_iterator)

        # Calculate SVs from eigen vectors of tridiag_matrix
        sv_matrix, s_values = pt_utils.calculate_svs(
            tridiag_matrix, input_vector_matrix
        )

        sv_matrix_store[i, :, :] = sv_matrix
        s_values_store[i, :] = s_values ** 2

    # Prepare plotting
    fig, axes = plt.subplots(nrows=2, ncols=2)

    # Do averaging
    sv_matrix_average = np.mean(sv_matrix_store, axis=0)
    s_values_average = np.mean(s_values_store, axis=0)

    sb.heatmap(np.abs(sv_matrix_average.T), ax=axes[0, 0])
    axes[0, 0].set_title("Lanczos singular vectors")

    sb.heatmap(np.abs(np_e_vectors.T), ax=axes[0, 1])
    axes[0, 1].set_title("Numpy singular vectors")

    gs = axes[1, 0].get_gridspec()

    for ax in axes[1, :]:
        ax.remove()

    axbig = fig.add_subplot(gs[1:, :])
    axbig.plot(np.abs(s_values_average), label="Lanczos eigen values")
    axbig.plot(np.abs(np_e_values), label="Numpys eigen values")
    axbig.legend()

    plt.suptitle(
        f"Compare Lanczos algorithm with np.linalg.eig\n on diagonal matrix | $n_runs$={n_runs}"
    )

    plt.show()


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    # verif_arg_setup = a_parsers.VerificationArgParser()
    # verif_arg_setup.setup_parser()
    # args = verif_arg_setup.args

    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        sh_utils.update_dependent_params(params)
        sh_utils.update_arrays(params)

    profiler.start()

    verify_lanczos_algorithm()

    profiler.stop()
    print(profiler.output_text(color=True))
