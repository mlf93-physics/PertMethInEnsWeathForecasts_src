import sys

sys.path.append("..")
import pathlib as pl
import itertools as it
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
import lorentz63_experiments.perturbations.normal_modes as pert_nm
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.importing.import_data_funcs as g_import
import general.plotting.plot_data as g_plt_data
import general.utils.argument_parsers as a_parsers
from general.params.model_licences import Models
from config import MODEL

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params


def plot_breed_vectors(args):

    # Import breed vectors
    breed_vector_units, _ = pt_import.import_breed_vectors(args)
    # Import info file
    pert_info_dict = g_import.import_info_file(
        pl.Path(args["path"], args["exp_folder"])
    )
    # Set number of vectors/profiles
    args["n_profiles"] = min(args["num_units"], breed_vector_units.shape[0])

    # Average and norm vectors
    mean_breed_vector_units = np.mean(breed_vector_units, axis=1)
    normed_mean_breed_vector_units = mean_breed_vector_units.T / np.reshape(
        np.linalg.norm(mean_breed_vector_units.T, axis=0), (1, args["n_profiles"])
    )

    # Calculate orthonormality
    orthonormality = [
        x.dot(y) for x, y in it.combinations(normed_mean_breed_vector_units.T, 2)
    ]
    orthonormality_matrix = np.zeros((args["n_profiles"], args["n_profiles"]))
    orthonormality_matrix[np.triu_indices(args["n_profiles"], k=1)] = np.abs(
        orthonormality
    )

    # Plotting
    plt.figure()
    plt.imshow(orthonormality_matrix, cmap="Reds")
    plt.xlabel("BV index, i")
    plt.ylabel("BV index, j")
    plt.title(
        "Orthogonality of BVs | Lorentz63 model \n"
        + f"$N_{{vectors}}$={args['n_profiles']}, $\\sigma={pert_info_dict['sigma']}$"
        + f", r={pert_info_dict['r_const']}, b={pert_info_dict['b_const']:.2f}"
    )
    plt.colorbar()

    plt.figure()
    plt.plot(normed_mean_breed_vector_units, "k")
    plt.xlabel("BV component index")
    plt.ylabel("BV component")
    plt.title(
        "Breed Vectors 2D | Lorentz63 model \n"
        + f"$N_{{vectors}}$={args['n_profiles']}, $\\sigma={pert_info_dict['sigma']}$"
        + f", r={pert_info_dict['r_const']}, b={pert_info_dict['b_const']:.2f}"
    )

    # Only make 3D plot if possible according to the dimension of the system
    if normed_mean_breed_vector_units.shape[0] == 3:
        origin = np.zeros(args["n_profiles"])

        plt.figure()
        ax3 = plt.axes(projection="3d")
        ax3.quiver(
            origin,
            origin,
            origin,
            normed_mean_breed_vector_units[0, :],
            normed_mean_breed_vector_units[1, :],
            normed_mean_breed_vector_units[2, :],
            # normalize=True,
            # length=0.5,
        )
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-1, 1)
        ax3.set_zlim(-1, 1)
        ax3.set_title(
            "Breed Vectors 3D | Lorentz63 model \n"
            + f"$N_{{vectors}}$={args['n_profiles']}, $\\sigma={pert_info_dict['sigma']}$"
            + f", r={pert_info_dict['r_const']}, b={pert_info_dict['b_const']:.2f}"
        )


def plot_breed_comparison_to_nm(args):
    # Import breed vectors
    breed_vector_units, breed_vec_header_dicts = pt_import.import_breed_vectors(args)
    # Import perturbation info file
    pert_info_dict = g_import.import_info_file(
        pl.Path(args["path"], args["exp_folder"])
    )

    # Average vectors for each unit
    mean_breed_vector_units = np.mean(breed_vector_units, axis=1)

    # Prepare arguments
    args["n_profiles"] = min(args["num_units"], breed_vector_units.shape[0])

    # Gather start times
    start_times = [
        breed_vec_header_dicts[i]["perturb_pos"] * params.stt
        for i in range(args["n_profiles"])
    ]
    # Add offset to start_times to reach end times
    start_times = [
        start_times[i] + pert_info_dict["n_cycles"] * pert_info_dict["time_per_cycle"]
        for i in range(args["n_profiles"])
    ]

    args["start_time"] = start_times

    # Import u_profiles at breed vector positions
    u_init_profiles, profile_positions, _ = g_import.import_start_u_profiles(args)

    # Find eigenvectors at breed vector end positions
    (
        e_vector_matrix,
        e_values_max,
        e_vector_collection,
        e_value_collection,
    ) = pert_nm.find_normal_modes(u_init_profiles, args, n_profiles=args["n_profiles"])

    normed_mean_breed_vector_units = mean_breed_vector_units.T / np.reshape(
        np.linalg.norm(mean_breed_vector_units.T, axis=0), (1, args["n_profiles"])
    )

    normed_e_vector_matrix = e_vector_matrix.real / np.reshape(
        np.linalg.norm(e_vector_matrix.real, axis=0), (1, args["n_profiles"])
    )

    plt.plot(normed_mean_breed_vector_units)
    plt.gca().set_prop_cycle(None)
    plt.plot(
        normed_e_vector_matrix,
        linestyle="--",
    )


def plot_breed_error_norm(args):
    g_plt_data.plot_error_norm_vs_time(args=args, normalize_start_time=False)


if __name__ == "__main__":
    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()

    args = vars(stand_plot_arg_parser.args)
    print("args", args)

    # Add missing arguments to make util funcs work
    args["burn_in_lines"] = 0

    if "vectors" in args["plot_type"]:
        plot_breed_vectors(args)
    elif "nm_compare" in args["plot_type"]:
        plot_breed_comparison_to_nm(args)
    elif "bv_error_norm" in args["plot_type"]:
        plot_breed_error_norm(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    if not args["noplot"]:
        plt.tight_layout()
        plt.show()
