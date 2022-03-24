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
import general.utils.util_funcs as g_utils
import general.utils.experiments.exp_utils as e_utils
import general.utils.importing.import_data_funcs as g_import
import general.utils.plot_utils as g_plt_utils
import general.plotting.plot_config as plt_config
import general.analyses.analyse_data as g_anal
import general.utils.argument_parsers as a_parsers
import general.utils.user_interface as g_ui
from libs.libutils import type_utils as lib_type_utils
import config as cfg


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
    num_subplot_cols = 3
    num_subplot_rows = math.ceil(len_folders / num_subplot_cols)

    # Define axis to plot on
    method_axis = {"rd": 0, "nm": 1, "bv": 2, "bv_eof": 3, "sv": 6, "lv": 9, "rf": 12}
    num_methods = 13

    # fig: plt.Figure = plt.figure()
    fig, axes = plt.subplots(
        ncols=num_subplot_cols,
        nrows=num_subplot_rows,
        sharex=True,
        sharey=True,
        # subplot_kw=dict(projection="3d"),
    )
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = [axes]

    # args["n_files"] = args["n_profiles"]

    max_exp_growth_rate = -np.inf
    min_exp_growth_rate = np.inf

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

        u_ref_stores = np.array(u_ref_stores)

        # Get indices of first run per profile
        run_in_profile_array = g_utils.get_values_from_dicts(
            header_dicts, "run_in_profile"
        )
        first_run_indices = np.where(
            np.array(run_in_profile_array, dtype=np.int32) == 0
        )[0]

        (
            _,
            profile_mean_growth_rates,
        ) = g_anal.execute_mean_exp_growth_rate_vs_time_analysis(
            args, u_stores, header_dicts=header_dicts, anal_type=args["exp_growth_type"]
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

        # Prepare info on perturbation
        folder_path = pl.Path(folder)
        digits_in_name = lib_type_utils.get_digits_from_string(folder_path.name)
        # Get perturb_type
        if digits_in_name is not None:
            perturb_type = folder_path.name.split(
                lib_type_utils.zpad_string(str(digits_in_name), n_zeros=2)
            )[0]
        else:
            perturb_type = folder_path.name.split("_")[0]

        if len(args["exp_folders"]) == num_methods:
            offset = digits_in_name if digits_in_name is not None else 0
            axis_index = method_axis[perturb_type] + offset
        else:
            axis_index = i

        scatter_plot = axes[axis_index].scatter(
            u_ref_stores[first_run_indices, 0, 0],
            u_ref_stores[first_run_indices, 0, 2],
            c=profile_mean_growth_rates[-1, :],
            # alpha=0.4,
            zorder=5,
            marker=".",
            s=4,
            norm=norm,
            cmap=cmap,
        )

        # Prepare titles
        # Take into account the _ in bv_eof
        perturb_type = perturb_type.replace("_", "-")
        if args["tolatex"]:
            if digits_in_name is not None:
                subtitle = f"$\\textnormal{{{perturb_type.upper()}}}^\\textnormal{{{digits_in_name + 1}}}$"
            else:
                subtitle = f"$\\textnormal{{{perturb_type.upper()}}}$"
        else:
            if digits_in_name is not None:
                subtitle = f"{perturb_type.upper()}{digits_in_name + 1}"
            else:
                subtitle = f"{perturb_type.upper()}"

        axes[axis_index].set_title(subtitle)

    # Remove leftover axes
    for j in range(len(axes) - 1, i, -1):
        axes[j].remove()  # Remove from figure
        axes = np.delete(axes, j)  # Remove from array

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header=f"{args['exp_growth_type'].capitalize()} exp. growth rate distribution",
        detailed=False,
        title_suffix=pl.Path(folder).parent.name,
    )
    if not args["tolatex"]:
        fig.suptitle(title)

    label_axes: plt.Axes = fig.add_subplot(111, frame_on=False)
    label_axes.tick_params(
        labelcolor="none", bottom=False, left=False, right=False, top=False
    )
    label_axes.set_ylabel("z")

    # Set x labels and ticks of buttom axes
    if len(args["exp_folders"]) == num_methods:
        for offset in range(1, 3):
            axes[method_axis["lv"] + offset].set_xlabel("x")
            axes[method_axis["lv"] + offset].xaxis.set_tick_params(labelbottom=True)

        axes[method_axis["rf"]].set_xlabel("x")

    # fig.subplots_adjust(bottom=0.2)
    # cbar_ax = fig.add_axes([0.15, 0.15, 0.05, 0.7])
    plt.subplots_adjust(
        top=0.956, bottom=0.08, left=0.08, right=0.995, hspace=0.6, wspace=0.13
    )
    cax = fig.add_axes([0.4, 0.12, 0.5, 0.05])
    fig.colorbar(
        scatter_plot,
        cax=cax,
        shrink=0.5,
        orientation="horizontal",
        label="$\\kappa(t)$"
        if args["exp_growth_type"].lower() == "instant"
        else "$\\kappa_{{mean}}(t_0)$",
    )

    if args["tolatex"]:
        plt_config.remove_legends(axes)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath="thesis_figures/results_and_analyses/l63/",
            file_name=f"compare_{args['exp_growth_type'].lower()}_exp_growth_rate_dists",
        )


def plot_pert_vectors3D(args: dict, axes: plt.Axes = None):
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(projection="3d")

    e_utils.update_compare_exp_folders(args)

    n_runs_per_profile_dict: dict = {
        "rd": 500,
        "nm": 500,
        "bv": 8,
        "bv_eof": 3,
        "sv": 3,
        "lv": 3,
        "rf": 500,
    }
    markerstyles_dict: dict = {
        "rd": ".",
        "nm": ".",
        "bv": ["o", "p"],
        "bv_eof": ["<", ">"],
        "sv": ["+", "x"],
        "lv": ["s", "p"],
        "rf": ".",
    }

    (
        perturbations_store,
        perturb_positions_store,
    ) = prepare_pert_vectors_for_compare_plot(
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
            color = METHOD_COLORS[mode]
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
                    linestyle=LINESTYLES[j] if mode != "bv" else LINESTYLES[0],
                    linewidth=1.5,
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
                facecolors=METHOD_COLORS[mode],
                color=METHOD_COLORS[mode],
                linewidth=0.1,
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

    # Import info on optimization time for SVs
    sv_vector_folder = list(
        filter(
            lambda folder: folder if pl.Path(folder).name == "sv_vectors" else None,
            args["exp_folders"],
        )
    )
    fsv_vector_folder = list(
        filter(
            lambda folder: folder if pl.Path(folder).name == "fsv_vectors" else None,
            args["exp_folders"],
        )
    )

    if len(sv_vector_folder) == 1:
        args["exp_folder"] = sv_vector_folder[0]
        sv_exp_setup = g_import.import_exp_info_file(args)
        sv_opt_samples = int(sv_exp_setup["integration_time"] * tts)
    elif len(sv_vector_folder) > 1:
        raise ValueError("Too many sv_vector folders found")
    else:
        sv_opt_samples = 20

    if len(fsv_vector_folder) == 1:
        args["exp_folder"] = fsv_vector_folder[0]
        fsv_exp_setup = g_import.import_exp_info_file(args)
        fsv_opt_samples = int(fsv_exp_setup["integration_time"] * tts)
    elif len(fsv_vector_folder) > 1:
        raise ValueError("Too many sv_vector folders found")
    else:
        fsv_opt_samples = 20

    # Plot perturbation start point
    axes.plot(
        u_data[dt_sample_offset, 0],
        u_data[dt_sample_offset, 1],
        u_data[dt_sample_offset, 2],
        "kx",
        label="_nolegend_",
    )

    axes.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    axes.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    axes.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    axes.grid(False)

    # Plot optimization time path of particle
    axes.plot(
        u_data[dt_sample_offset : dt_sample_offset + sv_opt_samples + 1, 0],
        u_data[dt_sample_offset : dt_sample_offset + sv_opt_samples + 1, 1],
        u_data[dt_sample_offset : dt_sample_offset + sv_opt_samples + 1, 2],
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
        title_suffix=f"time={perturb_positions_store[list(perturb_positions_store.keys())[0]][0] * stt:.2f}",
    )

    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_zlabel("z")
    axes.set_title(title)
    # Set limits according to rd data
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_zlim(zlim)
    # Set view
    axes.view_init(elev=args["elev"], azim=args["azim"])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.ticklabel_format(axis="z", style="sci", scilimits=(0, 0))
    axes.legend(loc="center right", bbox_to_anchor=(1.35, 0.5))

    if args["tolatex"]:
        plt_config.remove_legends(axes)
        axes.ticklabel_format(style="plain")
        plt_config.adjust_axes(axes)

    if args["save_fig"]:
        g_plt_utils.save_figure(
            args,
            subpath="thesis_figures/results_and_analyses/l63/",
            file_name=args["save_fig_name"],
        )
        # g_plt_utils.save_interactive_fig(
        #     fig,
        #     subpath="thesis_figures/results_and_analyses/l63/",
        #     name="pert_vectors_3D_interactive",
        # )


def plot_pert_vector_dists(args: dict, axes: plt.Axes = None):
    if axes is None:
        fig = plt.figure()
        axes = fig.add_subplot(projection="3d")

    e_utils.update_compare_exp_folders(args)

    n_runs_per_profile_dict: dict = {
        "bv_eof": 3,
        "sv": 3,
        "lv": 3,
    }

    (
        perturbations_store,
        perturb_positions_store,
    ) = prepare_pert_vectors_for_compare_plot(
        args,
        n_runs_per_profile_dict,
    )

    # Get first key in perturb_positions_store
    temp_key = list(perturb_positions_store.keys())[0]
    # Import from start of appropriate ref record to last perturb position
    first_pert_position = np.min(perturb_positions_store[temp_key])
    args["ref_start_time"] = (
        (first_pert_position * stt)
        // cfg.GLOBAL_PARAMS.record_max_time
        * cfg.GLOBAL_PARAMS.record_max_time
    )
    args["ref_end_time"] = (np.max(perturb_positions_store[temp_key]) + 1) * stt
    time, u_data, header_dict = g_import.import_ref_data(args=args)

    # Plot all perturbation vectors
    for i, item_tuple in enumerate(perturbations_store.items()):
        vector, perturbations = item_tuple
        # Scale up perturbation
        perturbations *= 200

        # Take into account if first position is not from first record
        perturb_pos_offset = int(
            first_pert_position
            // (cfg.GLOBAL_PARAMS.record_max_time * tts)
            * cfg.GLOBAL_PARAMS.record_max_time
            * tts
        )

        if vector in args["vectors"]:
            color = METHOD_COLORS[vector]
            for specific_run_index in range(n_runs_per_profile_dict[vector]):
                for j in range(args["n_profiles"]):
                    axes.quiver(
                        u_data[
                            perturb_positions_store[vector][j] - perturb_pos_offset, 0
                        ],
                        u_data[
                            perturb_positions_store[vector][j] - perturb_pos_offset, 1
                        ],
                        u_data[
                            perturb_positions_store[vector][j] - perturb_pos_offset, 2
                        ],
                        perturbations[
                            0, j * n_runs_per_profile_dict[vector] + specific_run_index
                        ],
                        perturbations[
                            1, j * n_runs_per_profile_dict[vector] + specific_run_index
                        ],
                        perturbations[
                            2, j * n_runs_per_profile_dict[vector] + specific_run_index
                        ],
                        label=vector
                        + "_"
                        + lib_type_utils.zpad_string(str(j), n_zeros=2),
                        zorder=10,
                        linestyle=LINESTYLES[specific_run_index],
                        linewidth=1.5,
                        # marker=markerstyles_dict[mode][j],
                        color=color,
                    )

        # Plot trajectory
        first_pos_rel_record = int(
            first_pert_position % (cfg.GLOBAL_PARAMS.record_max_time * tts)
        )
        axes.plot(
            u_data[first_pos_rel_record:, 0],
            u_data[first_pos_rel_record:, 1],
            u_data[first_pos_rel_record:, 2],
            "k-",
        )
        # Plot start point
        axes.plot(
            u_data[first_pos_rel_record, 0],
            u_data[first_pos_rel_record, 1],
            u_data[first_pos_rel_record, 2],
            "kx",
        )


def prepare_pert_vectors_for_compare_plot(
    args,
    n_runs_per_profile_dict,
):
    perturbations_store = {}
    perturb_positions_store = {}
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
        perturb_positions_store[args["vectors"][i]] = perturb_positions

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

    return perturbations_store, perturb_positions_store


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
        args,
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
    plt_config.adjust_default_fig_axes_settings(args)

    if "time_to_run" in args:
        args["Nt"] = int(args["time_to_run"] / dt * sample_rate)

    if "pert_vectors2D" in args["plot_type"]:
        plot_pert_vectors2D(args)
    elif "pert_vectors3D" in args["plot_type"]:
        plot_pert_vectors3D(args)
    elif "pert_vector_dists":
        plot_pert_vector_dists(args)
    elif "growth_rate_dist":
        plot_mean_exp_growth_rate_distribution(args)
    else:
        raise ValueError(f"No plot method present for plot_type={args['plot_type']}")

    g_plt_utils.save_or_show_plot(args)
