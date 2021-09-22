import sys

sys.path.append("..")
import numpy as np
from pyinstrument import Profiler
import lorentz63_experiments.perturbations.normal_modes as pert_nm
import general.utils.import_data_funcs as g_import

profiler = Profiler()


def analyse_normal_mode_dist(args):
    profiler.start()

    # Import reference data
    # time, u_data, ref_header_dict = g_import.import_ref_data(args=args)
    u_profiles, ref_header_dict = g_import.import_profiles_for_nm_analysis(args)

    (
        e_vector_matrix,
        e_vector_collection,
        e_value_collection,
    ) = pert_nm.find_normal_modes(u_profiles, args, n_profiles=u_profiles.shape[1])

    # Get maximum e_values
    e_values = np.array(e_value_collection)
    e_values = np.max(e_values, axis=1)

    profiler.stop()
    print(profiler.output_text())

    return u_profiles, e_values, e_vector_matrix, ref_header_dict
