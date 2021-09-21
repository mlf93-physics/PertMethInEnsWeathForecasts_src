import sys

sys.path.append("..")
import numpy as np
import lorentz63_experiments.perturbations.normal_modes as pert_nm
import general.utils.import_data_funcs as g_import


def analyse_normal_mode_dist(args):

    # Import reference data
    # time, u_data, ref_header_dict = g_import.import_ref_data(args=args)
    u_init_profiles, perturb_positions, header_dict = g_import.import_start_u_profiles(
        args=args
    )

    (
        e_vector_matrix,
        e_vector_collection,
        e_value_collection,
    ) = pert_nm.find_normal_modes(
        u_init_profiles, args, n_profiles=u_init_profiles.shape[1]
    )

    print("e_value_collection", e_value_collection)
