"""Make plots related to the breed vector calculations

Example
-------
python ../general/plotting/breed_vector_plotter.py
--plot_type=bv_error_norm
--exp_folder=pt_vectors/test1_after_params_refactor
--endpoint

"""

import sys

sys.path.append("..")
import pathlib as pl

import config as cfg
import general.analyses.breed_vector_eof_analysis as bv_analysis
import general.analyses.plot_analyses as g_plt_anal
import general.plotting.plot_data as g_plt_data
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.importing.import_utils as g_imp_utils
import general.utils.plot_utils as g_plt_utils
import general.plotting.plot_config as plt_config
import general.utils.user_interface as g_ui
import matplotlib.pyplot as plt
import numpy as np
from general.params.model_licences import Models
from general.utils.module_import.type_import import *
from mpl_toolkits import mplot3d
from pyinstrument import Profiler

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.plotting.plot_data as sh_plot
    import shell_model_experiments.utils.util_funcs as sh_utils
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType

    params = PAR_SH
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
    import lorentz63_experiments.plotting.plot_data as l63_plot

    params = l63_params


def plot_breed_vectors(args):

    # Import breed vectors
    breed_vector_units, _, _, _, _ = pt_import.import_perturb_vectors(
        args, raw_perturbations=True
    )

    # Import info file
    pert_info_dict = g_import.import_info_file(
        pl.Path(
            args["datapath"],
            args["pert_vector_folder"],
            args["exp_folder"],
            "perturb_data",
        )
    )
    # Set number of vectors/profiles
    args["n_profiles"] = min(args["n_units"], breed_vector_units.shape[0])

    # Average and norm vectors
    mean_breed_vector_units = np.mean(breed_vector_units, axis=1)

    normed_mean_breed_vector_units = mean_breed_vector_units / np.repeat(
        np.reshape(
            np.linalg.norm(mean_breed_vector_units, axis=1),
            (args["n_profiles"], 1),
        ),
        repeats=params.sdim,
        axis=1,
    )

    # Calculate orthonormality
    orthonormality_matrix = g_plt_anal.orthogonality_of_vectors(
        normed_mean_breed_vector_units
    )

    ortho_bv_title = g_plt_utils.generate_title(
        args,
        header_dict=pert_info_dict,
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
        args,
        header_dict=pert_info_dict,
        title_header="Breed Vectors 2D | Lorentz63 model \n",
        title_suffix=f"$N_{{vectors}}$={args['n_profiles']}",
    )

    plt.figure()
    plt.plot(normed_mean_breed_vector_units.T, "k")
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
            normed_mean_breed_vector_units[:, 0],
            normed_mean_breed_vector_units[:, 1],
            normed_mean_breed_vector_units[:, 2],
            # normalize=True,
            # length=0.5,
        )

        bv_3d_title = g_plt_utils.generate_title(
            args,
            header_dict=pert_info_dict,
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
    (
        breed_vector_units,
        _,
        _,
        _,
        breed_vec_header_dicts,
    ) = pt_import.import_perturb_vectors(args)

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
    if args["ylim"] is not None:
        axes[0].set_ylim(args["ylim"][0], args["ylim"][1])

    start_time, end_time = g_imp_utils.get_start_end_times_from_exp_setup(
        exp_setup, pert_info_dict
    )

    args["ref_start_time"] = start_time
    args["ref_end_time"] = end_time

    if cfg.MODEL == Models.SHELL_MODEL:
        sh_plot.plot_energy(args, axes=axes[1])
    elif cfg.MODEL == Models.LORENTZ63:
        l63_plot.plot_energy(args, axes=axes[1])


def plot_breed_eof_vectors_average(args: dict, axes: plt.Axes = None):
    # Prepare plot_kwargs
    plot_kwargs: dict = {
        "xlabel": "$i$",
        "ylabel": "$n$",
        "title_header": "Averaged BV-EOFs",
        "vector_label": "$\\langle|e_{n,i}| \\rangle$",
    }

    g_plt_data.plot2D_average_vectors(
        args,
        axes=axes,
        characteristic_value_name="$s_i^2$",
        plot_kwargs=plot_kwargs,
    )

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath="thesis_figures/" + args["save_sub_folder"],
            file_name="average_bv_eof_vectors_with_variances",
        )


def plot_breed_vectors_average(args: dict, axes: plt.Axes = None):
    # Generate cmap
    cmap, norm = g_plt_utils.get_custom_cmap(
        vcenter=0, neg_thres=0.4, pos_thres=0.6, cmap_handle=plt.cm.bwr
    )

    # Prepare plot_kwargs
    plot_kwargs: dict = {
        "xlabel": "$i$",
        "ylabel": "$n$",
        "title_header": "Averaged BVs rel. mean BV",
        "vector_label": "$\\langle|b_{n,i} - \\langle b_{n,i} \\rangle_{(i,t)}| \\rangle$",
        # "cmap": cmap,
    }

    g_plt_data.plot2D_average_vectors(
        args,
        axes=axes,
        rel_mean_vector=True,
        plot_kwargs=plot_kwargs,
        no_char_values=True,
    )

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath="thesis_figures/" + args["save_sub_folder"],
            file_name="average_bv_vectors",
        )


def plot_breed_eof_vectors_3D(args: dict):
    """Plot the EOF breed vectors in 3D

    Parameters
    ----------
    args : dict
        Run-time arguments
    """

    perturb_vectors, _, _, _, perturb_header_dicts = pt_import.import_perturb_vectors(
        args
    )

    print("\nCalculating BV-EOF vectors from BV vectors")
    eof_vectors, variances = bv_analysis.calc_eof_vectors(perturb_vectors)
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
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    args = stand_plot_arg_parser.args

    g_ui.confirm_run_setup(args)

    # Shell model specific
    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        sh_utils.update_dependent_params(params)
        sh_utils.set_params(params, parameter="sdim", value=args["sdim"])
        sh_utils.update_arrays(params)

    plt_config.adjust_default_fig_axes_settings(args)

    # Make profiler
    profiler = Profiler()
    profiler.start()

    if "pert_vectors" in args["plot_type"]:
        plot_breed_vectors(args)
    elif "nm_compare" in args["plot_type"]:
        plot_breed_comparison_to_nm(args)
    elif "bv_error_norm" in args["plot_type"]:
        plot_breed_error_norm(args)
    elif "bv_vectors_average" in args["plot_type"]:
        plot_breed_vectors_average(args)
    elif "bv_eof_vectors_average" in args["plot_type"]:
        plot_breed_eof_vectors_average(args)
    elif "bv_eof_3D" in args["plot_type"]:
        plot_breed_eof_vectors_3D(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    profiler.stop()
    print(profiler.output_text(color=True))
    g_plt_utils.save_or_show_plot(args)
