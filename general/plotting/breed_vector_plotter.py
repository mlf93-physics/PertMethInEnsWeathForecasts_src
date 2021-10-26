import sys

sys.path.append("..")
import pathlib as pl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import shell_model_experiments.params as sh_params
import shell_model_experiments.plotting.plot_data as sh_plot
import lorentz63_experiments.params.params as l63_params
import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
import lorentz63_experiments.plotting.plot_data as l63_plot
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.importing.import_data_funcs as g_import
import general.plotting.plot_data as g_plt_data
import general.utils.plot_utils as g_plt_utils
import general.analyses.plot_analyses as g_plt_anal
import general.analyses.breed_vector_eof_analysis as bv_analysis
import general.utils.argument_parsers as a_parsers
import general.utils.user_interface as g_ui
from general.params.model_licences import Models
from config import MODEL

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params


def plot_breed_vectors(args):

    # Import breed vectors
    breed_vector_units, _ = pt_import.import_perturb_vectors(args)

    # Import info file
    pert_info_dict = g_import.import_info_file(
        pl.Path(args["datapath"], args["exp_folder"])
    )
    # Set number of vectors/profiles
    args["n_profiles"] = min(args["n_units"], breed_vector_units.shape[0])

    # Average and norm vectors
    mean_breed_vector_units = np.mean(breed_vector_units, axis=1)
    normed_mean_breed_vector_units = mean_breed_vector_units / np.reshape(
        np.linalg.norm(mean_breed_vector_units, axis=0), (1, args["n_profiles"])
    )

    # Calculate orthonormality
    orthonormality_matrix = g_plt_anal.orthogonality_of_vectors(
        normed_mean_breed_vector_units
    )

    ortho_bv_title = g_plt_utils.generate_title(
        pert_info_dict,
        args,
        title_header="Orthogonality of BVs | Lorentz63 model \n",
        title_suffix=f"$N_{{vectors}}$={args['n_profiles']}",
    )

    # Plotting
    plt.figure()
    plt.imshow(orthonormality_matrix, cmap="Reds")
    plt.xlabel("BV index, i")
    plt.ylabel("BV index, j")
    plt.title(ortho_bv_title)
    plt.colorbar()

    bv_2d_title = g_plt_utils.generate_title(
        pert_info_dict,
        args,
        title_header="Breed Vectors 2D | Lorentz63 model \n",
        title_suffix=f"$N_{{vectors}}$={args['n_profiles']}",
    )

    plt.figure()
    plt.plot(normed_mean_breed_vector_units, "k")
    plt.xlabel("BV component index")
    plt.ylabel("BV component")
    plt.title(bv_2d_title)

    # Only make 3D plot if possible according to the dimension of the system
    if normed_mean_breed_vector_units.shape[1] == 3:
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

        bv_3d_title = g_plt_utils.generate_title(
            pert_info_dict,
            args,
            title_header="Breed Vectors 3D | Lorentz63 model \n",
            title_suffix=f"$N_{{vectors}}$={args['n_profiles']}",
        )

        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        ax3.set_xlim(-1, 1)
        ax3.set_ylim(-1, 1)
        ax3.set_zlim(-1, 1)
        ax3.set_title(bv_3d_title)


def plot_breed_comparison_to_nm(args):
    # Import breed vectors
    breed_vector_units, breed_vec_header_dicts = pt_import.import_perturb_vectors(args)

    # Import perturbation info file
    pert_info_dict = g_import.import_info_file(
        pl.Path(args["datapath"], args["exp_folder"])
    )

    # Average vectors for each unit
    mean_breed_vector_units = np.mean(breed_vector_units, axis=1)

    # Prepare arguments
    args["n_profiles"] = min(args["n_units"], breed_vector_units.shape[0])

    # Gather start times
    start_times = [
        breed_vec_header_dicts[i]["perturb_pos"] * params.stt
        for i in range(args["n_profiles"])
    ]
    # Add offset to start_times to reach end times
    start_times = [
        start_times[i] + pert_info_dict["n_cycles"] * pert_info_dict["integration_time"]
        for i in range(args["n_profiles"])
    ]

    args["start_times"] = start_times

    # Import u_profiles at breed vector positions
    u_init_profiles, profile_positions, _ = g_import.import_start_u_profiles(args)

    # Find eigenvectors at breed vector end positions
    (
        e_vector_matrix,
        e_values_max,
        e_vector_collection,
        e_value_collection,
    ) = l63_nm_estimator.find_normal_modes(
        u_init_profiles, args, n_profiles=args["n_profiles"]
    )

    normed_mean_breed_vector_units = mean_breed_vector_units.T / np.reshape(
        np.linalg.norm(mean_breed_vector_units.T, axis=0), (1, args["n_profiles"])
    )

    normed_e_vector_matrix = e_vector_matrix.real / np.reshape(
        np.linalg.norm(e_vector_matrix.real, axis=0), (1, args["n_profiles"])
    )
    orthogonality = normed_e_vector_matrix.T @ normed_mean_breed_vector_units

    plt.plot(normed_mean_breed_vector_units)
    plt.gca().set_prop_cycle(None)
    plt.plot(
        normed_e_vector_matrix,
        linestyle="--",
    )

    plt.figure()
    plt.imshow(orthogonality, cmap="Reds")
    plt.xlabel("BV index")
    plt.ylabel("NM index")


def plot_breed_error_norm(args):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    exp_setup = g_import.import_exp_info_file(args)

    # Add subfolder
    args["exp_folder"] = pl.Path(args["exp_folder"], "perturb_data")
    # Import perturbation info file
    pert_info_dict = g_import.import_info_file(
        pl.Path(args["datapath"], args["exp_folder"])
    )

    g_plt_data.plot_error_norm_vs_time(
        args=args, normalize_start_time=False, axes=axes[0], exp_setup=exp_setup
    )

    # Set limits
    axes[0].set_ylim(args["ylim"][0], args["ylim"][1])

    # Prepare ref import
    if "start_times" in exp_setup:
        start_time = exp_setup["start_times"][0]
        end_time = (
            exp_setup["start_times"][0]
            + exp_setup["n_cycles"] * exp_setup["integration_time"]
        )
    elif "eval_times" in exp_setup:
        # Adjust start- and endtime differently depending on if only last
        # perturbation data is saved, or all perturbation data is saved.
        if pert_info_dict["save_last_pert"]:
            start_time = exp_setup["eval_times"][0] - exp_setup["integration_time"]
            end_time = (
                start_time + pert_info_dict["n_units"] * exp_setup["integration_time"]
            )
        else:
            start_time = (
                exp_setup["eval_times"][0]
                - exp_setup["n_cycles"] * exp_setup["integration_time"]
            )
            end_time = exp_setup["eval_times"][0]
    else:
        raise ValueError("start_time could not be determined from exp setup")

    args["ref_start_time"] = start_time
    args["ref_end_time"] = end_time

    if MODEL == Models.SHELL_MODEL:
        sh_plot.plots_related_to_energy(args, axes=axes[1])
    elif MODEL == Models.LORENTZ63:
        l63_plot.plot_energy(args, axes=axes[1])


def plot_breed_eof_vectors_3D(args: dict):
    """Plot the EOF breed vectors in 3D

    Parameters
    ----------
    args : dict
        Run-time arguments
    """

    perturb_vectors, perturb_header_dicts = pt_import.import_perturb_vectors(args)

    eof_vectors = bv_analysis.calc_bv_eof_vectors(perturb_vectors)
    n_vectors: int = eof_vectors.shape[2]

    origin: np.ndarray = np.zeros(n_vectors)

    plt.figure()
    ax3 = plt.axes(projection="3d")
    ax3.quiver(
        origin,
        origin,
        origin,
        eof_vectors[0, 0, :],
        eof_vectors[0, 1, :],
        eof_vectors[0, 2, :],
    )

    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_zlim(-1, 1)


if __name__ == "__main__":
    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    args = stand_plot_arg_parser.args

    g_ui.confirm_run_setup(args)

    if "pert_vector_folder" in args["plot_type"]:
        plot_breed_vectors(args)
    elif "nm_compare" in args["plot_type"]:
        plot_breed_comparison_to_nm(args)
    elif "bv_error_norm" in args["plot_type"]:
        plot_breed_error_norm(args)
    elif "bv_eof_3D" in args["plot_type"]:
        plot_breed_eof_vectors_3D(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    g_plt_utils.save_or_show_plot(args)
