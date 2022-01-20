import sys

sys.path.append("..")
from pyinstrument import Profiler
import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
import general.utils.importing.import_perturbation_data as pt_import

profiler = Profiler()


def analyse_normal_mode_dist(args):
    profiler.start()

    # Import reference data
    u_profiles, ref_header_dict = pt_import.import_profiles_for_nm_analysis(args)

    (
        e_vector_matrix,
        e_values_max,
        e_vector_collection,
        e_value_collection,
    ) = l63_nm_estimator.find_normal_modes(
        u_profiles, args, n_profiles=u_profiles.shape[1]
    )

    profiler.stop()
    print(profiler.output_text(color=True))

    return u_profiles, e_values_max, e_vector_matrix, ref_header_dict
