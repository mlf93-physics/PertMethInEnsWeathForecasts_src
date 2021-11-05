import sys

sys.path.append("..")
import numpy as np
import shell_model_experiments.params as sh_params
import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
import lorentz63_experiments.params.params as l63_params
import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
from general.params.model_licences import Models
import config as cfg

# Get jacobian calculator for model
if cfg.MODEL == Models.SHELL_MODEL:
    init_jacobian = sh_nm_estimator.init_jacobian
    calc_jacobian = sh_nm_estimator.calc_jacobian
    params = sh_params
elif cfg.MODEL == Models.LORENTZ63:
    init_jacobian = l63_nm_estimator.init_jacobian
    calc_jacobian = l63_nm_estimator.calc_jacobian
    params = l63_params


def lyapunov_analyser(u_data):
    n_profiles = u_data.shape[0]

    # NOTE - Avoid this transformation by editing the shape of u_init_profiles all other places
    u_data = u_data.T

    j_matrix_old = np.ones((params.sdim, params.sdim), dtype=np.float64)

    j_matrix_new = init_jacobian(args)
    for i in range(n_profiles):
        calc_jacobian(j_matrix_new, u_data[:, i], r_const=args["r_const"])
        j_matrix_old = j_matrix_old @ j_matrix_new

    e_values, e_vectors = np.linalg.eig(j_matrix_old)

    print("e_values, e_vectors", e_values, e_vectors)
    print("e_values, e_vectors", e_values ** (1 / n_profiles), e_vectors)


def run_analysis(args):

    time, u_data, ref_header_dict = g_import.import_ref_data(args=args)

    lyapunov_analyser(u_data)


if __name__ == "__main__":
    # Get arguments
    _stand_arg_setup = a_parsers.StandardArgSetup()
    _stand_arg_setup.setup_parser()
    _standard_model_arg_setup = a_parsers.StandardModelArgSetup()
    _standard_model_arg_setup.setup_parser()
    _ref_arg_setup = a_parsers.ReferenceAnalysisArgParser()
    _ref_arg_setup.setup_parser()
    args = _ref_arg_setup.args

    print("args", args)

    run_analysis(args)
