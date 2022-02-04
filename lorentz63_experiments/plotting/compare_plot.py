"""Plotting functions relevant for the perturbations of the Lorentz63 model experiments

Example
-------
python plotting/plot_perturbations.py --plot_type=<>

"""
import sys

sys.path.append("..")
import pathlib as pl
import math
import matplotlib.pyplot as plt
from lorentz63_experiments.params.params import *
import general.utils.arg_utils as a_utils
from general.utils.module_import.type_import import *
from general.plotting.plot_params import *
import general.utils.running.runner_utils as r_utils
import general.utils.experiments.exp_utils as e_utils
import general.utils.importing.import_data_funcs as g_import
import general.utils.plot_utils as g_plt_utils
import general.analyses.analyse_data as g_anal
import general.utils.argument_parsers as a_parsers
import general.utils.user_interface as g_ui
from libs.libutils import type_utils as lib_type_utils
import config as cfg

cfg.GLOBAL_PARAMS.record_max_time = 3000


def plot_mean_exp_growth_rate_distribution(args: dict):
    """Plot the distribution of the exp growth rates across the L63 attractor

    Due to the way the import of data is set up, you need to run with sv or bv_eof
    preceeding any other perturbations, i.e. -pt sv rd instead of -pt rd

    Parameters
    ----------
    args : dict
        Run-time arguments
    """

    e_utils.update_compare_exp_folders(args)

    len_folders = len(args["exp_folders"])
    num_subplot_rows = math.floor(len_folders / 2) + 1
    num_subplot_cols = math.ceil(len_folders / num_subplot_rows)

    # fig: plt.Figure = plt.figure()
    fig, axes = plt.subplots(
        num_subplot_cols,
        num_subplot_rows,
        subplot_kw=dict(projection="3d"),
    )
    axes = axes.ravel()

    # args["n_files"] = args["n_profiles"]

    max_exp_growth_rate = -np.inf
    min_exp_growth_rate = np.inf
    save_u_ref_stores = None

    for i, folder in enumerate(args["exp_folders"]):
        # Set exp_folder
        args["exp_folder"] = folder

        (
            u_stores,
            perturb_time_pos_list,
            perturb_time_pos_list_legend,
            header_dicts,
            u_ref_stores,
        ) = g_import.import_perturbation_velocities(
            args, search_pattern="*perturb*.csv"
        )
        if i == 0:
            save_u_ref_stores = np.array(u_ref_stores)

        (
            _,
            profile_mean_growth_rates,
        ) = g_anal.execute_mean_exp_growth_rate_vs_time_analysis(
            args, u_stores, header_dicts=header_dicts, anal_type="mean"
        )

        temp_max_exp_growth_rate = np.max(profile_mean_growth_rates[-1, :])
        temp_min_exp_growth_rate = np.min(profile_mean_growth_rates[-1, :])
        max_exp_growth_rate = (
            temp_max_exp_growth_rate
            if temp_max_exp_growth_rate > max_exp_growth_rate
            else max_exp_growth_rate
        )
        min_exp_growth_rate = (
            temp_min_exp_growth_rate
            if temp_min_exp_growth_rate < min_exp_growth_rate
            else min_exp_growth_rate
        )

        # Prepare cmap and norm
        cmap, norm = g_plt_utils.get_custom_cmap(
            vmin=min_exp_growth_rate,
            vmax=max_exp_growth_rate,
            vcenter=(max_exp_growth_rate + min_exp_growth_rate) / 2,
            cmap_handle=plt.cm.jet,
        )

        scatter_plot = axes[i].scatter(
            save_u_ref_stores[:, 0, 0],
            save_u_ref_stores[:, 0, 1],
            save_u_ref_stores[:, 0, 2],
            c=profile_mean_growth_rates[-1, :],
            alpha=0.4,
            zorder=5,
            marker=".",
            norm=norm,
            cmap=cmap,
        )
        axes[i].set_title(pl.Path(folder).name.split("_perturbations")[0])
        axes[i].xaxis.set_ticklabels([])
        axes[i].yaxis.set_ticklabels([])
        axes[i].zaxis.set_ticklabels([])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(scatter_plot, cax=cbar_ax)

    # Remove leftover axes
    for j in range(i + 1, len(axes)):
        axes[j].remove()

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header="Mean exp. growth rate distribution",
        detailed=False,
        title_suffix=pl.Path(folder).parent.name,
    )
    fig.suptitle(title)


def plot_pert_vectors3D(args: dict, axes: plt.Axes = None):
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(projection="3d")

    e_utils.update_compare_exp_folders(args)

    # Get colors
    color_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    n_runs_per_profile_dict: dict = {
        "rd": 500,
        "nm": 500,
        "bv": 8,
        "bv_eof": 3,
        "sv": 3,
        "rf": 500,
    }
    markerstyles_dict: dict = {
        "rd": ".",
        "nm": ".",
        "bv": ["o", "p"],
        "bv_eof": ["<", ">"],
        "sv": ["+", "x"],
        "rf": ".",
    }

    perturbations_store, perturb_positions = prepare_pert_vectors_for_compare_plot(
        args,
        n_runs_per_profile_dict,
    )
    # Import reference data
    dt = 50
    args["ref_start_time"] = args["start_times"][0] - dt
    args["ref_end_time"] = args["start_times"][0] + dt
    time, u_data, header_dict = g_import.import_ref_data(args=args)
    dt_sample_offset = int(dt * tts)

    # Plot all perturbation vectors
    for i, item_tuple in enumerate(perturbations_store.items()):
        mode, perturbations = item_tuple
        # Scale up perturbation
        perturbations *= 300

        if mode in args["vectors"]:
            color = color_list[i]
            for j in range(n_runs_per_profile_dict[mode]):
                axes.quiver(
                    u_data[dt_sample_offset, 0],
                    u_data[dt_sample_offset, 1],
                    u_data[dt_sample_offset, 2],
                    perturbations[0, j],
                    perturbations[1, j],
                    perturbations[2, j],
                    label=mode + "_" + lib_type_utils.zpad_string(str(j), n_zeros=2),
                    zorder=10,
                    linestyle=LINESTYLES[j],
                    linewidth=2,
                    # marker=markerstyles_dict[mode][j],
                    color=color,
                )
        else:

            # Add reference
            perturbations += u_data[dt_sample_offset, :][:, np.newaxis]

            if mode == "rd":
                xmin = np.min(perturbations[0, :])
                xmax = np.max(perturbations[0, :])
                ymin = np.min(perturbations[1, :])
                ymax = np.max(perturbations[1, :])
                zmin = np.min(perturbations[2, :])
                zmax = np.max(perturbations[2, :])
                xrange = xmax - xmin
                yrange = ymax - ymin
                zrange = zmax - zmin
                range_percent = 0.1
                xlim: tuple = (
                    xmin - xrange * range_percent,
                    xmax + xrange * range_percent,
                )
                ylim: tuple = (
                    ymin - yrange * range_percent,
                    ymax + yrange * range_percent,
                )
                zlim: tuple = (
                    zmin - zrange * range_percent,
                    zmax + zrange * range_percent,
                )

            axes.scatter(
                perturbations[0, :],
                perturbations[1, :],
                perturbations[2, :],
                label=mode,
                alpha=0.4,  # if mode in ["rd", "nm", "rf"] else 1.0,
                zorder=5,
                marker=markerstyles_dict[mode],
                facecolors=color_list[i],
                color=color_list[i],
            )

    # Plot attractor
    axes.plot(
        u_data[:, 0],
        u_data[:, 1],
        u_data[:, 2],
        "k-",
        alpha=0.6,
        label="Reference",
        linewidth=0.5,
        zorder=0,
    )

    # Plot perturbation start point
    axes.plot(
        u_data[dt_sample_offset, 0],
        u_data[dt_sample_offset, 1],
        u_data[dt_sample_offset, 2],
        "kx",
        label="_nolegend_",
    )

    # Plot preceding path of particle
    part_dt = 20
    axes.plot(
        u_data[dt_sample_offset - part_dt : dt_sample_offset + 1, 0],
        u_data[dt_sample_offset - part_dt : dt_sample_offset + 1, 1],
        u_data[dt_sample_offset - part_dt : dt_sample_offset + 1, 2],
        "k-",
        alpha=1,
        label="Particle path",
        linewidth=2,
        zorder=1,
    )

    title = g_plt_utils.generate_title(
        args,
        title_header="Perturbation Vectors",
        detailed=False,
        title_suffix=f"time={perturb_positions[0] * stt:.2f}",
    )

    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.set_title(title)
    # Set limits according to rd data
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_zlim(zlim)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="z", style="sci", scilimits=(0, 0))
    axes.legend(loc="center right", bbox_to_anchor=(1.35, 0.5))


def prepare_pert_vectors_for_compare_plot(
    args,
    n_runs_per_profile_dict,
):
    perturbations_store = {}
    vector_folders = [folder for folder in args["exp_folders"] if "vectors" in folder]

    # Import perturb vectors
    for i, folder in enumerate(vector_folders):
        raw_perturbations = True

        args["exp_folder"] = folder
        # Update arguments
        args["pert_mode"] = args["vectors"][i]
        args["n_runs_per_profile"] = n_runs_per_profile_dict[args["vectors"][i]]

        perturbations, perturb_positions, _ = r_utils.prepare_perturbations(
            args, raw_perturbations=raw_perturbations
        )

        perturbations_store[args["vectors"][i]] = perturbations

    # Filter out already imported perturbations
    filtered_perturbations = [
        item for item in args["perturbations"] if item not in args["vectors"]
    ]
    # Generate all other perturbation vectors
    for i, mode in enumerate(filtered_perturbations):
        args["pert_mode"] = mode

        args["n_runs_per_profile"] = n_runs_per_profile_dict[mode]
        args["start_times"] = [perturb_positions[0] * stt]

        perturbations, perturb_positions, _ = r_utils.prepare_perturbations(
            args, raw_perturbations=raw_perturbations
        )
        perturbations_store[mode] = perturbations

    return perturbations_store, perturb_positions


def plot_pert_vectors2D(args, axes: plt.Axes = None):

    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1)

    e_utils.update_compare_exp_folders(args)

    n_runs_per_profile_dict: dict = {
        "rd": 500,
        "nm": 500,
        "bv": 2,
        "bv_eof": 2,
        "sv": 2,
        "rf": 500,
    }
    markerstyles_dict: dict = {
        "rd": ".",
        "nm": ".",
        "bv": ["o", "p"],
        "bv_eof": ["<", ">"],
        "sv": ["+", "x"],
        "rf": ".",
    }

    vector_folders = [folder for folder in args["exp_folders"] if "vectors" in folder]

    # Import perturb vectors and plot
    for i, folder in enumerate(vector_folders):
        raw_perturbations = True

        args["exp_folder"] = folder
        # Update arguments
        args["pert_mode"] = args["vectors"][i]
        args["n_runs_per_profile"] = n_runs_per_profile_dict[args["vectors"][i]]

        perturbations, perturb_positions, _ = r_utils.prepare_perturbations(
            args, raw_perturbations=raw_perturbations
        )

        color = None
        for j in range(n_runs_per_profile_dict[args["vectors"][i]]):
            scatterplot = axes.scatter(
                perturbations[0, j],
                perturbations[1, j],
                label=args["vectors"][i]
                + "_"
                + lib_type_utils.zpad_string(str(j), n_zeros=2),
                zorder=10,
                marker=markerstyles_dict[args["vectors"][i]][j],
                color=color,
            )
            color = scatterplot.get_facecolors()[0].tolist()

    # Filter out already plotted perturbations
    filtered_perturbations = [
        item for item in args["perturbations"] if item not in args["vectors"]
    ]
    # Generate all other perturbation vectors
    for i, mode in enumerate(filtered_perturbations):
        args["pert_mode"] = mode

        args["n_runs_per_profile"] = n_runs_per_profile_dict[mode]
        args["start_times"] = [perturb_positions[0] * stt]

        perturbations, perturb_positions, _ = r_utils.prepare_perturbations(
            args, raw_perturbations=raw_perturbations
        )
        if mode == "rd":
            xmin = np.min(perturbations[0, :])
            xmax = np.max(perturbations[0, :])
            ymin = np.min(perturbations[1, :])
            ymax = np.max(perturbations[1, :])
            xrange = xmax - xmin
            yrange = ymax - ymin
            range_percent = 0.1
            xlim: tuple = (xmin - xrange * range_percent, xmax + xrange * range_percent)
            ylim: tuple = (ymin - yrange * range_percent, ymax + yrange * range_percent)

        axes.scatter(
            perturbations[0, :],
            perturbations[1, :],
            label=mode,
            alpha=0.6,  # if mode in ["rd", "nm", "rf"] else 1.0,
            zorder=0,
            marker=markerstyles_dict[mode],
        )

    # Import reference data
    dt = 0.1
    args["ref_start_time"] = args["start_times"][0] - dt
    args["ref_end_time"] = args["start_times"][0] + dt
    time, u_data, header_dict = g_import.import_ref_data(args=args)
    # Get reference data rel start_time
    dt_sample_offset = int(dt * tts)
    rel_u_data = u_data - u_data[dt_sample_offset, :]

    axes.plot(rel_u_data[:, 0], rel_u_data[:, 1], "k-", label="Reference")
    axes.plot(
        rel_u_data[dt_sample_offset, 0],
        rel_u_data[dt_sample_offset, 1],
        "kx",
        label="_nolegend_",
    )

    title = g_plt_utils.generate_title(
        args,
        title_header="Perturbation Vectors",
        detailed=False,
        title_suffix=f"time={perturb_positions[0] * stt:.2f}",
    )

    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_title(title)
    # Set limits according to rd data
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes.legend(loc="center right", bbox_to_anchor=(1.35, 0.5))

    g_plt_utils.save_figure(
        subpath=pl.Path(
            "lorentz63_experiments/compare_perturbations/perturbation_vectors_offset0.01_series/"
        ),
        file_name=f"compare_perturbation_vectors_time{perturb_positions[0] * stt:.2f}",
        fig=fig,
    )


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    compare_plot_arg_parser = a_parsers.ComparisonPlottingArgParser()
    compare_plot_arg_parser.setup_parser()
    args: dict = compare_plot_arg_parser.args

    a_utils.react_on_comparison_arguments(args)

    # Initiate arrays
    # initiate_sdim_arrays(args["sdim"])
    g_ui.confirm_run_setup(args)

    if "time_to_run" in args:
        args["Nt"] = int(args["time_to_run"] / dt * sample_rate)

    if "pert_vectors2D" in args["plot_type"]:
        plot_pert_vectors2D(args)
    elif "pert_vectors3D" in args["plot_type"]:
        plot_pert_vectors3D(args)
    elif "growth_rate_dist":
        plot_mean_exp_growth_rate_distribution(args)
    else:
        raise ValueError(f"No plot method present for plot_type={args['plot_type']}")

    g_plt_utils.save_or_show_plot(args)
