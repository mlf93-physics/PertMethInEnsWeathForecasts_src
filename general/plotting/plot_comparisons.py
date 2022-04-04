"""Make plots to compare the different perturbation methods

Example
-------
python ../general/plotting/plot_comparisons.py
--plot_type=error_norm_compare
--exp_folder=test1_new_params

"""

import sys

sys.path.append("..")
import math
import re
import pathlib as pl

import config as cfg
import general.plotting.plot_data as g_plt_data
import general.utils.argument_parsers as a_parsers
import general.utils.arg_utils as a_utils
import general.utils.importing.import_data_funcs as g_import
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.importing.import_utils as g_imp_utils
import general.utils.plot_utils as g_plt_utils
import general.analyses.analyse_data as g_a_data
import general.analyses.plot_analyses as g_plt_anal
from general.plotting.plot_params import *
import general.plotting.plot_config as plt_config
from general.utils.module_import.type_import import *
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
import general.utils.experiments.exp_utils as e_utils
import general.utils.running.runner_utils as r_utils
from libs.libutils import type_utils as lib_type_utils
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import seaborn as sb
from general.params.model_licences import Models

# Get parameters for model
if cfg.MODEL == Models.SHELL_MODEL:
    import shell_model_experiments.plotting.plot_data as sh_plot
    import shell_model_experiments.utils.util_funcs as sh_utils
    import shell_model_experiments.utils.special_params as sh_sparams
    import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator
    from shell_model_experiments.params.params import PAR as PAR_SH
    from shell_model_experiments.params.params import ParamsStructType

    params = PAR_SH
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams
    import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator
    import lorentz63_experiments.plotting.plot_data as l63_plot

    params = l63_params
    sparams = l63_sparams


def plt_pert_components(args: dict, axes: plt.Axes = None):

    if len(args["vectors"]) == 0:
        raise ValueError("--vectors argument mandatory for this plot")

    e_utils.update_compare_exp_folders(args)

    # Prepare axes
    if axes is None:
        axes = plt.axes()

    shell_index = np.log2(params.k_vec_temp)
    cmap_list_base = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # ipert = 0

    vector_folders = [folder for folder in args["exp_folders"] if "vectors" in folder]

    # Import perturb vectors and plot
    for i, folder in enumerate(vector_folders):
        folder_path = pl.Path(folder)
        if re.match(fr"bv(\d+_vectors|_vectors)", folder_path.name):
            raw_perturbations = False
        else:
            raw_perturbations = True

        args["exp_folder"] = folder
        (
            vector_units,
            _,
            _,
            eval_pos,
            header_dicts,
        ) = pt_import.import_perturb_vectors(args, raw_perturbations=raw_perturbations)

        mean_vectors = np.mean(np.abs(vector_units), axis=0).T
        norm_mean_vectors = g_utils.normalize_array(mean_vectors, norm_value=1, axis=0)

        # if "bv" in folder:
        #     anal_file = g_utils.get_analysis_file(args, type="velocities")
        #     mean_u_data, ref_header_dict = g_import.import_data(
        #         file_name=anal_file, start_line=1
        #     )

        #     norm_mean_u_data = g_utils.normalize_array(
        #         np.abs(mean_u_data.T), norm_value=1, axis=0
        #     )
        #     norm_mean_vectors /= norm_mean_u_data
        #     norm_mean_vectors = g_utils.normalize_array(
        #         norm_mean_vectors, norm_value=1, axis=0
        #     )

        # Prepare cmap for SVs
        cmap = g_plt_utils.set_color_cycle_for_vectors(
            axes,
            vector_type=folder_path.name.split("_vectors")[0],
            n_vectors=norm_mean_vectors.shape[1],
        )

        vector_lines = axes.plot(
            shell_index,
            norm_mean_vectors,
            linestyle="solid",
            # color=cmap_list[ipert],
        )

        vector_lines[0].set_label(
            str(pl.Path(args["exp_folder"]).name).split("_vectors")[0]
            + f" ({cmap.name.strip('_r')})"
        )

        # ipert += 1

    # Generate all other perturbation vectors
    vector_units_list = []
    vector_label_list = []

    args["start_times"] = np.array(eval_pos) * params.stt

    for item in args["perturbations"]:
        # Get the requested perturbations
        if item == "rf":
            args["pert_mode"] = "rf"
            vector_label_list.append("rf")
        elif item == "rd":
            args["pert_mode"] = "rd"
            vector_label_list.append("rd")

        elif item == "nm":
            args["pert_mode"] = "nm"
            vector_label_list.append("nm")
        else:
            continue

        perturbations, _, exec_all_runs_per_profile = r_utils.prepare_perturbations(
            args, raw_perturbations=True
        )
        vector_units_list.append(perturbations[sparams.u_slice, :])

    axes.set_prop_cycle("color", cmap_list_base[6:])

    for i, vector_units in enumerate(vector_units_list):
        mean_vectors = np.mean(np.abs(vector_units), axis=1)
        norm_mean_vectors = g_utils.normalize_array(mean_vectors, norm_value=1, axis=0)

        vector_lines = axes.plot(
            shell_index,
            norm_mean_vectors,
            linestyle="solid",
        )
        vector_lines[0].set_label(vector_label_list[i])
        # ipert += 1

    axes.xaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True))

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header="Mean perturbation vectors",
        title_suffix=f"$N_{{vectors}}$={args['n_profiles']}",
        detailed=False,
    )

    axes.set_xlabel("Shell index, i")
    axes.set_ylabel("$\\langle|v_i|\\rangle$")
    axes.set_yscale("log")
    axes.set_ylim(1e-4, 1)
    axes.set_title(title)
    axes.legend()


def import_multiple_vector_dirs(
    args,
    raw_perturbations=True,
    force_no_ref_import=True,
    retrieve_header_key: str = "",
    dtype=np.float64,
):

    header_values = {}
    if len(retrieve_header_key) > 0:
        header_values = {retrieve_header_key: []}

    all_vector_units = np.zeros(
        (
            len(args["exp_folders"]),
            args["n_profiles"],
            args["n_runs_per_profile"],
            params.sdim,
        ),
        dtype=sparams.dtype,
    )

    for i, folder in enumerate(args["exp_folders"]):
        args["exp_folder"] = folder
        (vector_units, _, _, _, vec_header_dicts,) = pt_import.import_perturb_vectors(
            args,
            raw_perturbations=raw_perturbations,
            dtype=dtype,
            force_no_ref_import=force_no_ref_import,
        )

        if len(retrieve_header_key) > 0:
            header_values[retrieve_header_key].append(
                vec_header_dicts[0][retrieve_header_key]
            )

        all_vector_units[i, :, :, :] = vector_units

    return all_vector_units, header_values


def plt_vec_compared_to_lv(args, axes: plt.Axes = None, pair_vectors=False):

    save_exp_folder = args["exp_folder"]
    save_vectors: List[str] = args["vectors"]
    retrieve_header_key = "time_to_run"

    # Initialise dicts
    vector_folder_units_dict: dict = {}
    header_values_dict: dict = {}
    orthogonality_dict: dict = {}  # Orthogonality rel LV
    adj_orthogonality_dict: dict = {}  # Orthogonality rel adj LV or forward LV
    mean_vector_lv_orthogonality_dict = {}
    mean_vector_adj_lv_orthogonality_dict = {}

    # Determine lv range
    if args["lvs_to_compare"] is not None:
        lv_range = [*args["lvs_to_compare"]]
        n_lvs = len(lv_range)
    elif args["n_lvs_to_compare"] is not None:
        lv_range = range(args["n_lvs_to_compare"])
        n_lvs = args["n_lvs_to_compare"]

    for vector in save_vectors:
        # Reset exp_folder
        args["exp_folder"] = save_exp_folder
        # Get folder units
        vector_folder_units, header_values = prepare_vector_folder_units(
            args,
            vector=vector,
            raw_perturbations=True,
            force_no_ref_import=True,
            dtype=np.complex128
            if "sv" in vector
            else sparams.dtype,  # Both sv's and fsv's
            retrieve_header_key=retrieve_header_key,
            zero_iw_only=(vector == "lv" or vector == "alv")
            and "fsv" not in save_vectors,
        )
        vector_folder_units_dict[vector] = vector_folder_units
        header_values_dict[vector] = header_values

        if vector not in ["lv", "alv"]:
            orthogonality_dict[vector] = np.empty(
                (
                    n_lvs,
                    len(header_values[retrieve_header_key]),
                    args["n_profiles"],
                    args["n_runs_per_profile"],
                ),
                dtype=sparams.dtype,
            )
            adj_orthogonality_dict[vector] = np.copy(orthogonality_dict[vector])

    # Get first key (vector) in header_values_dict not being lv or alv
    header_val_dict_keys: list = list(header_values_dict.keys())
    first_key: str = list(filter(lambda x: "lv" not in x, header_val_dict_keys))[0]
    iw_values: List[float] = sorted(header_values_dict[first_key][retrieve_header_key])

    # Calculate orthogonality between vectors, i.e. between all BVs, SVs and LVs (multiple LV)
    for n, lv_index in enumerate(lv_range):
        for i, value in enumerate(iw_values):
            for j in range(args["n_profiles"]):
                for k, vector in enumerate(save_vectors):
                    if vector in ["lv", "alv"]:
                        continue

                    if vector in ["bv", "sv"]:
                        if "lv" in save_vectors:
                            orthogonality_dict[vector][
                                n, i, j, :
                            ] = g_plt_anal.orthogonality_to_vector(
                                vector_folder_units_dict["lv"][
                                    0, j, lv_index, :
                                ],  # 0 in first index is for using only LVs evaluated at time 0
                                vector_folder_units_dict[vector][i, j, :, :],
                            )
                        if "alv" in save_vectors:
                            adj_orthogonality_dict[vector][
                                n, i, j, :
                            ] = g_plt_anal.orthogonality_to_vector(
                                vector_folder_units_dict["alv"][
                                    0, j, lv_index, :
                                ],  # 0 in first index is for using only LV evaluated at time 0
                                vector_folder_units_dict[vector][i, j, :, :],
                            )

                    if vector in ["fsv"]:
                        if "lv" in save_vectors:
                            orthogonality_dict[vector][
                                n, i, j, :
                            ] = g_plt_anal.orthogonality_to_vector(
                                vector_folder_units_dict["lv"][
                                    i + 1, j, lv_index, :
                                ],  # +1 in order to skip lv valid at time 0
                                vector_folder_units_dict[vector][i, j, :, :],
                            )
                        if "alv" in save_vectors:
                            adj_orthogonality_dict[vector][
                                n, i, j, :
                            ] = g_plt_anal.orthogonality_to_vector(
                                vector_folder_units_dict["alv"][
                                    i + 1, j, lv_index, :
                                ],  # +1 in order to skip alv valid at time 0
                                vector_folder_units_dict[vector][i, j, :, :],
                            )

    # Average orthogonality
    for vector in save_vectors:
        if vector in ["lv", "alv"]:
            continue

        if vector in ["fsv", "sv"]:
            if "lv" in save_vectors:
                mean_vector_lv_orthogonality_dict[vector] = np.mean(
                    np.abs(orthogonality_dict[vector]), axis=2
                )
            if "alv" in save_vectors:
                mean_vector_adj_lv_orthogonality_dict[vector] = np.mean(
                    np.abs(adj_orthogonality_dict[vector]), axis=2
                )
        if vector in ["bv"]:
            if "lv" in save_vectors:
                mean_vector_lv_orthogonality_dict[vector] = np.mean(
                    np.mean(np.abs(orthogonality_dict[vector]), axis=3), axis=2
                )
            if "alv" in save_vectors:
                mean_vector_adj_lv_orthogonality_dict[vector] = np.mean(
                    np.mean(np.abs(adj_orthogonality_dict[vector]), axis=3), axis=2
                )

    # Prepare axes
    if axes is None:
        axes: plt.Axes = plt.axes()

    # Plot orthogonality vs iw
    for vector in save_vectors:
        # Plot BVs vs LVs
        if vector in ["bv"]:
            # Get cmap
            # cmap_list, _ = g_plt_utils.get_non_repeating_colors(n_colors=n_lvs)
            cmap_list = sb.color_palette()
            axes.set_prop_cycle("color", cmap_list)
            vector_vs_lines: list = []
            if "lv" in save_vectors:
                vector_vs_lv_lines: List[plt.Line2D] = axes.plot(
                    iw_values,
                    mean_vector_lv_orthogonality_dict[vector].T,
                )
                vector_vs_lines.append(vector_vs_lv_lines)
            if "alv" in save_vectors:
                vector_vs_alv_lines = axes.plot(
                    iw_values,
                    mean_vector_adj_lv_orthogonality_dict[vector].T,
                    linestyle="dashed",
                )
                vector_vs_lines.append(vector_vs_alv_lines)

            line_objs = zip(*vector_vs_lines)

            # Set labels and colors
            for i, line_obj in enumerate(line_objs):
                if "lv" in save_vectors:
                    line_obj[0].set_label(
                        f"{vector.upper()} vs LV{i}",
                    )
                if "alv" in save_vectors:
                    if len(line_obj) == 1:
                        line_obj[0].set_label(
                            f"{vector.upper()} vs ALV{i}",
                        )
                        # Reset linestyle
                        line_obj[0].set_linestyle("solid")
                    else:
                        line_obj[1].set_label(
                            f"{vector.upper()} vs ALV{i}",
                        )
                        line_obj[1].set_color(line_obj[0].get_color())

        # Plot SVs/FSVs vs LVs
        if "sv" in vector:
            # Get cmap
            # cmap_list, _ = g_plt_utils.get_non_repeating_colors(
            #     n_colors=args["n_runs_per_profile"]
            # )
            cmap_list = sb.color_palette()
            axes.set_prop_cycle("color", cmap_list)
            for n, lv_index in enumerate(lv_range):
                vector_vs_lines: list = []
                if "lv" in save_vectors:
                    if pair_vectors:
                        vector_vs_lines.append(
                            axes.plot(
                                iw_values,
                                mean_vector_lv_orthogonality_dict[vector][n, :, n],
                                color=cmap_list[n],
                            )
                        )
                    else:
                        vector_vs_lv_lines = axes.plot(
                            iw_values,
                            mean_vector_lv_orthogonality_dict[vector][n, :, :],
                        )
                        vector_vs_lines.append(vector_vs_lv_lines)
                if "alv" in save_vectors:
                    if pair_vectors:
                        vector_vs_lines.append(
                            axes.plot(
                                iw_values,
                                mean_vector_adj_lv_orthogonality_dict[vector][n, :, n],
                                linestyle="dashed",
                                color=cmap_list[n],
                            )
                        )
                    else:
                        vector_vs_alv_lines = axes.plot(
                            iw_values,
                            mean_vector_adj_lv_orthogonality_dict[vector][n, :, :],
                            linestyle="dashed",
                        )
                        vector_vs_lines.append(vector_vs_alv_lines)

                line_objs = zip(*vector_vs_lines)

                # Set labels and colors
                vector_string: str = (
                    ("I" + vector.upper()) if vector == "sv" else vector.upper()
                )

                for i, line_obj in enumerate(line_objs):

                    if "lv" in save_vectors:
                        line_obj[0].set_label(
                            f"{vector_string}{n if pair_vectors else i} vs LV{lv_index}",
                        )
                    if "alv" in save_vectors:
                        if len(line_obj) == 1:
                            line_obj[0].set_label(
                                f"{vector_string}{n if pair_vectors else i} vs ALV{lv_index}",
                            )
                            # Reset linestyle
                            line_obj[0].set_linestyle("solid")
                        else:
                            line_obj[1].set_label(
                                f"{vector_string}{n if pair_vectors else i} vs ALV{lv_index}",
                            )
                            line_obj[1].set_color(line_obj[0].get_color())

    axes.set_xlabel("$t_{{OPT}}$")
    axes.set_ylabel("Absolute projectibility")
    axes.legend()

    title = g_plt_utils.generate_title(
        args,
        # header_dict=lv_vec_header_dicts[0],
        title_header="Projectibility vs IW",
        title_suffix=f"$n_{{units}}$={args['n_profiles']}",
        detailed=False,
    )
    axes.set_title(title)


def prepare_vector_folder_units(
    args,
    vector: str = "bv",
    raw_perturbations=True,
    force_no_ref_import=True,
    dtype=sparams.dtype,
    retrieve_header_key: str = "",
    zero_iw_only: bool = False,
):

    args["vectors"] = [vector]
    e_utils.update_compare_exp_folders(args)

    if zero_iw_only:
        args["exp_folders"] = [args["exp_folders"][0]]

    # Import data
    vector_folder_units, header_values = import_multiple_vector_dirs(
        args,
        raw_perturbations=raw_perturbations,
        force_no_ref_import=force_no_ref_import,
        retrieve_header_key=retrieve_header_key,
        dtype=dtype,
    )

    return vector_folder_units, header_values


def plt_BV_LYAP_vector_comparison(args):
    """Plot a comparison between breed and lyapunov vectors

    Parameters
    ----------
    args : dict
        Run-time arguments
    """
    # Setup number of units
    # args["n_units"] = 1

    # First folder: breed vectors
    args["exp_folder"] = args["exp_folders"][0]

    (
        breed_vector_units,
        _,
        _,
        _,
        breed_vec_header_dicts,
    ) = pt_import.import_perturb_vectors(args, raw_perturbations=False)
    # breed_vector_units = np.squeeze(breed_vector_units, axis=0)
    # Normalize vectors
    breed_vector_units = g_utils.normalize_array(
        breed_vector_units, norm_value=1, axis=2
    )

    # Second folder: lyapunov vectors
    args["exp_folder"] = args["exp_folders"][1]
    (
        lyapunov_vector_units,
        _,
        _,
        _,
        lyapunov_vec_header_dicts,
    ) = pt_import.import_perturb_vectors(args, raw_perturbations=True)

    # lyapunov_vector_units = np.squeeze(lyapunov_vector_units, axis=0)
    # Normalize vectors
    lyapunov_vector_units = g_utils.normalize_array(
        lyapunov_vector_units, norm_value=1, axis=2
    )

    n_vectors = args["n_runs_per_profile"]

    if args["n_profiles"] > 1:
        num_subplot_cols = math.floor(args["n_profiles"] / 2)
        num_subplot_rows = math.ceil(args["n_profiles"] / num_subplot_cols)
    else:
        num_subplot_cols = 1
        num_subplot_rows = 1

    # fig1, axes1 = plt.subplots(
    #     num_subplot_rows, num_subplot_cols, sharex=True, sharey=True
    # )
    fig2, axes2 = plt.subplots(
        num_subplot_rows, num_subplot_cols, sharex=True, sharey=True
    )

    if args["n_profiles"] == 1:
        # axes1 = np.array(axes1)
        axes2 = np.array(axes2)

    # axes1 = axes1.ravel()
    axes2 = axes2.ravel()

    # Add cbar axes
    cbar_ax = fig2.add_axes([0.91, 0.3, 0.03, 0.4])
    mean_orthogonality = np.empty(args["n_profiles"], dtype=np.float64)

    for i in range(args["n_profiles"]):
        # cmap_list, _ = g_plt_utils.get_non_repeating_colors(n_colors=n_vectors)
        # axes1[i].set_prop_cycle("color", cmap_list)
        # axes1[i].plot(lyapunov_vector_units[i, :, :].T.real, "--")

        # # Reset color cycle
        # axes1[i].set_prop_cycle("color", cmap_list)
        # axes1[i].plot(breed_vector_units[i, :, :].T.real, "-")
        # axes1[i].set_title(f"unit {i}")

        orthogonality_matrix = np.abs(
            breed_vector_units[i, :, :].conj() @ lyapunov_vector_units[i, :, :].T
        )

        mean_orthogonality[i] = np.mean(orthogonality_matrix)

        # print(
        #     "orthogonality_matrix mean",
        #     np.mean(
        #         orthogonality_matrix[~np.eye(orthogonality_matrix.shape[0], dtype=bool)]
        #     ),
        # )

        sb.heatmap(
            orthogonality_matrix,
            cmap="Reds",
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            ax=axes2[i],
            cbar=i == 0,
            cbar_ax=None if i else cbar_ax,
            annot_kws={"fontsize": 8},
        )
        axes2[i].invert_yaxis()
        axes2[i].set_xlabel("Lyapunov vector index")
        axes2[i].set_ylabel("Breed vector index")
        axes2[i].set_title(f"unit {i}")

    print("mean_orthogonality", np.mean(mean_orthogonality))

    # fig1.suptitle(f"Breed(-)/Lyapunov(--) vectors")
    fig2.suptitle(f"Orthogonality between\nBreed/Lyapunov vectors")
    # fig2.tight_layout(rect=[0, 0, 0.9, 1])


def plot_tlm_validity(args: dict, axes=None):
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)

    args["endpoint"] = True

    e_utils.update_compare_exp_folders(args)

    cmap_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    line_counter = 0
    u_stores = []
    for _, folder in enumerate(args["exp_folders"]):
        folder_path = pl.Path(folder)
        # Set exp_folder
        args["exp_folder"] = folder

        # digits_in_name = lib_type_utils.get_digits_from_string(folder_path.name)
        # if digits_in_name is not None:
        #     if isinstance(digits_in_name, int):
        #         perturb_type = folder_path.name.split(
        #             lib_type_utils.zpad_string(str(digits_in_name), n_zeros=2)
        #         )[0]
        #         color = METHOD_COLORS[perturb_type]

        #         linestyle = LINESTYLES[digits_in_name]

        # else:
        #     perturb_type = folder_path.name.split("_")[0]
        #     color = METHOD_COLORS[perturb_type]
        #     linestyle = None

        (
            u_store,
            perturb_time_pos_list,
            perturb_time_pos_list_legend,
            header_dicts,
            u_ref_stores,
        ) = g_import.import_perturbation_velocities(
            args,
            search_pattern="*perturb*.csv",
        )

        # Get number of runs per profile and n_profiles
        n_runs_per_profile = int(header_dicts[0]["n_runs_per_profile"])
        n_profiles = int(header_dicts[0]["n_profiles"])
        n_perturbations = len(perturb_time_pos_list)

        u_stores.append(u_store)

    u_diff_stores = []
    for i in range(n_perturbations):
        u_diff_stores.append(u_stores[1][i] - u_stores[0][i])

    (
        error_norm_vs_time,
        error_norm_mean_vs_time,
    ) = g_a_data.analyse_error_norm_vs_time(u_diff_stores, args=args)

    time_array = np.linspace(
        0,
        header_dicts[0]["time_to_run"],
        int(header_dicts[0]["time_to_run"] * params.tts) + args["endpoint"] * 1,
        dtype=np.float64,
        endpoint=args["endpoint"],
    )

    mean_of_logged_error_norm = np.mean(np.log(error_norm_vs_time[1:, :]), axis=1)

    plt.plot(time_array[1:], mean_of_logged_error_norm, "k")


def plot_error_norm_comparison(
    args: dict, axes=None, specific_runs_per_profile_dict=None
):
    """Plots a comparison of the error norm based in several different
    perturbation techniques

    Parameters
    ----------
    args : dict
        Run-time arguments
    """
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)

    axes = [axes, axes.twinx()]

    args["endpoint"] = True

    e_utils.update_compare_exp_folders(
        args, specific_runs_per_profile_dict=specific_runs_per_profile_dict
    )

    cmap_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    line_counter = 0
    for _, folder in enumerate(args["exp_folders"]):
        folder_path = pl.Path(folder)
        # Set exp_folder
        args["exp_folder"] = folder

        digits_in_name = lib_type_utils.get_digits_from_string(folder_path.name)
        if digits_in_name is not None:
            if isinstance(digits_in_name, int):
                perturb_type = folder_path.name.split(
                    lib_type_utils.zpad_string(str(digits_in_name), n_zeros=2)
                )[0]
                color = METHOD_COLORS[perturb_type]

                linestyle = LINESTYLES[digits_in_name]

        else:
            perturb_type = folder_path.name.split("_")[0]
            color = METHOD_COLORS[perturb_type]
            linestyle = None

        g_plt_data.plot_error_norm_vs_time(
            args,
            axes=axes[0],
            cmap_list=[color],
            linestyle=linestyle,
            linewidth=LINEWIDTHS["thin"],
            legend_on=False,
            normalize_start_time=False,
            plot_args=[],
        )
        lines: list = list(axes[0].get_lines())
        lines[line_counter].set_label(str(pl.Path(folder).name))

        len_lines = len(lines)
        line_counter += len_lines - line_counter

    axes[0].legend()

    # Import perturbation and experiment info files of last perturbation folder
    pert_info_dict = g_import.import_info_file(
        pl.Path(args["datapath"], args["exp_folder"])
    )
    exp_setup = g_import.import_exp_info_file(args)

    start_time, end_time = g_imp_utils.get_start_end_times_from_exp_setup(
        exp_setup, pert_info_dict
    )

    args["ref_start_time"] = start_time
    args["ref_end_time"] = end_time

    if cfg.MODEL == Models.SHELL_MODEL:
        sh_plot.plot_energy(
            args,
            axes=axes[1],
            plot_args=[],
        )
    elif cfg.MODEL == Models.LORENTZ63:
        l63_plot.plot_velocities(args, axes=axes[1])

    axes[0].set_zorder(10)
    axes[0].patch.set_visible(False)

    if args["tolatex"]:
        axes[0].get_legend().remove()
        plt_config.adjust_axes(axes)

    return axes


def plot_RMSE_and_spread_comparison(args: dict):
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True)

    args["endpoint"] = True

    e_utils.update_compare_exp_folders(args)

    cmap_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    line_counter = 0
    perturb_type_old = ""
    color_counter = 0
    perturb_type_u_store = []
    len_folders = len(args["exp_folders"])
    pert_type_change_counter = 0

    if not args["rmse_spread"]:
        cmap_list, _ = g_plt_utils.get_non_repeating_colors(n_colors=len_folders)
    else:
        cmap_list = None

    for i, folder in enumerate(args["exp_folders"]):
        folder_path = pl.Path(folder)
        perturb_type_w_digits = folder_path.name.split("_perturbations")[0]
        perturb_type = "".join(i for i in perturb_type_w_digits if not i.isdigit())

        # Try to get next folder
        try:
            next_folder_path = pl.Path(args["exp_folders"][i + 1])
            next_perturb_type = next_folder_path.name.split("_perturbations")[0]
            next_perturb_type = "".join(i for i in next_perturb_type if not i.isdigit())
        except IndexError:
            next_folder_path = None
            next_perturb_type = None

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

        perturb_type_u_store.append(u_stores)
        if args["rmse_spread"]:
            if next_perturb_type != perturb_type or i + 1 == len_folders:
                perturb_type_u_store_array = np.array(
                    perturb_type_u_store, dtype=sparams.dtype
                )
            else:
                continue
        else:
            perturb_type_u_store_array = np.array(
                perturb_type_u_store, dtype=sparams.dtype
            )

        # plt.plot(np.linalg.norm(perturb_type_u_store_array[0, 14, :, :], axis=1))

        n_profiles = np.unique(perturb_time_pos_list).size

        if perturb_type_u_store_array.shape[1] != n_profiles:
            # Reshape array to separate profiles
            perturb_type_u_store_array = np.reshape(
                perturb_type_u_store_array,
                (-1, n_profiles, int(header_dicts[0]["N_data"]), params.sdim),
            )
        # plt.plot(np.linalg.norm(perturb_type_u_store_array[4, 1, :, :], axis=1))
        # plt.show()

        perturb_type_u_store = []

        g_plt_data.plot_RMSE_and_spread(
            perturb_type_u_store_array,
            args,
            header_dict=header_dicts[0],
            axes=axes,
            label=perturb_type_w_digits if args["rmse_spread"] else perturb_type,
            color=cmap_list[i] if cmap_list is not None else None,
        )

    #     lines: list = list(axes.get_lines())
    #     lines[line_counter].set_label(str(pl.Path(folder).name))

    #     len_lines = len(lines)
    #     line_counter += len_lines - line_counter

    # axes.legend()


def plot_exp_growth_rate_comparison(args: dict, axes: plt.Axes = None):
    """Plots a comparison of the exponential growth rates vs time for the different
    perturbation methods

    Parameters
    ----------
    args : dict
        Run-time arguments
    """
    # args["endpoint"] = True

    if cfg.MODEL == cfg.Models.SHELL_MODEL:
        specific_runs_per_profile_dict = {
            "bv": None,
            "rd": None,
            "nm": None,
            "rf": None,
            "bv_eof": [0, 4, 9, 14],
            "sv": [0, 4, 9, 14],
            "lv": [1, 4, 9, 14],
        }
    else:
        specific_runs_per_profile_dict = None

    e_utils.update_compare_exp_folders(
        args, specific_runs_per_profile_dict=specific_runs_per_profile_dict
    )
    # Update number of folders after filtering
    len_folders = len(args["exp_folders"])

    cmap_list = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # cmap_list, _ = g_plt_utils.get_non_repeating_colors(
    #     n_colors=args["n_runs_per_profile"]  # , vmin=0.2, vmax=0.8
    # )
    # cmap_list[0] = "k"
    standalone_plot = False
    if axes is None:
        axes = plt.axes()
        standalone_plot = True

    perturb_type_old = ""
    color_counter = 0
    for i, folder in enumerate(args["exp_folders"]):
        folder_path = pl.Path(folder)

        digits_in_name = lib_type_utils.get_digits_from_string(folder_path.name)
        if digits_in_name is not None:
            if digits_in_name >= args["n_runs_per_profile"]:
                continue
            if isinstance(digits_in_name, int):
                perturb_type = folder_path.name.split(
                    lib_type_utils.zpad_string(str(digits_in_name), n_zeros=2)
                )[0]
                color = METHOD_COLORS[perturb_type]

                # linestyle = LINESTYLES[digits_in_name]
                linestyle = METHOD_LINESTYLES[perturb_type][i]
        else:
            perturb_type = folder_path.name.split("_")[0]
            color = METHOD_COLORS[perturb_type]
            linestyle = None

        # Set exp_folder
        args["exp_folder"] = folder

        # if i == 0:
        #     zorder = 10
        # else:
        #     zorder = 0
        g_plt_data.plot_exp_growth_rate_vs_time(
            args=args,
            axes=axes,
            color=color,  # cmap_list[i],
            # zorder=zorder,
            linestyle=linestyle,
            anal_type=args["exp_growth_type"],
            plot_args=[],
            title_suffix=str(folder_path.parent),
        )

    if cfg.MODEL == Models.LORENTZ63:
        lower_bound: float = -6
        upper_bound: float = 4
        axes.set_ylim(lower_bound, upper_bound)
        spacing: float = 2
        axes.set_yticks(
            np.linspace(
                lower_bound,
                upper_bound,
                int(abs(upper_bound - lower_bound) / spacing) + 1,
                endpoint=True,
            ),
            minor=False,
        )

    if standalone_plot:
        if args["tolatex"]:
            plt_config.remove_legends(axes)
            plt_config.adjust_axes(axes)

        if args["save_fig"]:
            if cfg.MODEL == Models.SHELL_MODEL:
                subfolder = "shell"
            elif cfg.MODEL == Models.LORENTZ63:
                subfolder = "l63"

            g_plt_utils.save_figure(
                args,
                subpath="thesis_figures/results_and_analyses/" + subfolder,
                file_name="compare_instant_exp_growth_rates",
            )


def plot_characteristic_value_vs_time(args: dict, axes: plt.Axes = None):
    if len(args["vectors"]) == 0:
        raise ValueError("--vectors argument mandatory for this plot")

    e_utils.update_compare_exp_folders(args)

    # Prepare axes
    if axes is None:
        axes = plt.axes()

    vector_folders = [folder for folder in args["exp_folders"] if "vectors" in folder]

    # Import perturb vectors and plot
    for i, folder in enumerate(vector_folders):
        folder_path = pl.Path(folder)
        if re.match(fr"bv(\d+_vectors|_vectors)", folder_path.name):
            raw_perturbations = False
        else:
            raw_perturbations = True

        args["exp_folder"] = folder
        (
            _,
            characteristic_values,
            u_init_profiles,
            eval_pos,
            header_dicts,
        ) = pt_import.import_perturb_vectors(
            args,
            raw_perturbations=raw_perturbations,
            dtype=np.complex128 if "sv" in folder_path.name else None,
        )

        # print(
        #     "s values",
        #     1 / header_dicts[0]["time_to_run"] * np.sqrt(characteristic_values),
        # )

        normed_characteristic_values = characteristic_values / np.max(
            characteristic_values
        )  # g_utils.normalize_array(
        #     characteristic_values, norm_value=1, axis=1
        # )

        array_to_plot = prepare_char_values_to_plot(args, normed_characteristic_values)

        # Only make eval_times array once
        if i == 0:
            eval_times = np.array(eval_pos, dtype=np.float64) * params.stt

        vector_type = folder_path.name.split("_vectors")[0]
        subplot_routine_characteristic_value_vs_time(
            axes, array_to_plot, eval_times, vector_type
        )

    if "nm" in args["perturbations"]:
        # Prepare u_init_profiles
        u_init_profiles = u_init_profiles[
            :,
            np.s_[
                0 : args["n_profiles"]
                * args["n_runs_per_profile"] : args["n_runs_per_profile"]
            ],
        ]

        # Get eigen values
        if cfg.MODEL == Models.SHELL_MODEL:
            _, _, e_value_collection = sh_nm_estimator.find_normal_modes(
                u_init_profiles,
                args,
                dev_plot_active=False,
                local_ny=header_dicts[0]["ny"],
            )
        elif cfg.MODEL == Models.LORENTZ63:
            _, _, _, e_value_collection = l63_nm_estimator.find_normal_modes(
                u_init_profiles,
                args,
                n_profiles=args["n_profiles"],
            )

        e_value_collection = np.array(e_value_collection, dtype=np.complex128)
        # kolm_sinai_entropy = sh_utils.get_kolm_sinai_entropy(e_value_collection, axis=1)
        # cumsum_char_values = np.cumsum(e_value_collection.real, axis=1)
        normed_characteristic_values = (
            e_value_collection.real
            / np.max(e_value_collection.real)
            # / kolm_sinai_entropy[:, np.newaxis]
        )

        array_to_plot = prepare_char_values_to_plot(
            args, normed_characteristic_values[:, : args["n_runs_per_profile"]]
        )

        # nm_axes = axes.twinx()
        subplot_routine_characteristic_value_vs_time(
            axes,
            array_to_plot,
            eval_times,
            vector_type="nm",
        )
        # nm_axes.set_ylabel("Cummulative char. value")
        # nm_axes.set_ylim(0, 1)
        # nm_axes.legend()

    axes.set_xlabel("Time")
    axes.set_ylabel("Normalized char. value")
    axes.set_yscale("log")

    title = g_plt_utils.generate_title(
        args,
        header_dict=header_dicts[0],
        title_header="Characteristic values vs time",
        title_suffix=f"display: {args['display_type']} real",
        detailed=False,
    )
    axes.set_title(title)
    axes.legend()

    return eval_times


def subplot_routine_characteristic_value_vs_time(
    axes, array_to_plot, eval_times, vector_type="sv"
):
    # Prepare cmap for vectors
    _ = g_plt_utils.set_color_cycle_for_vectors(
        axes,
        vector_type=vector_type,
        n_vectors=array_to_plot.shape[1],
    )

    lines = axes.plot(
        eval_times,
        array_to_plot,
    )
    lines[0].set_linestyle("dashed")
    lines[1].set_label(vector_type)
    lines[-1].set_linestyle("dashed")


def prepare_char_values_to_plot(args, normed_characteristic_values):
    if args["display_type"] == "all":
        array_to_plot: np.ndarray = normed_characteristic_values.real
    elif args["display_type"] == "quantile":
        quantiled_char_values = np.quantile(
            normed_characteristic_values, [1 / 4, 1 / 2, 3 / 4], axis=1
        )
        array_to_plot = np.empty((args["n_profiles"], 5), dtype=np.float64)
        array_to_plot[:, 0] = normed_characteristic_values[:, 0].real
        array_to_plot[:, 1] = quantiled_char_values[2, :].real
        array_to_plot[:, 2] = quantiled_char_values[1, :].real
        array_to_plot[:, 3] = quantiled_char_values[0, :].real
        array_to_plot[:, 4] = normed_characteristic_values[:, -1].real
    elif args["display_type"] == "specific":
        array_to_plot = np.empty((args["n_profiles"], 3), dtype=np.float64)
        array_to_plot[:, 0] = normed_characteristic_values[:, 0].real
        array_to_plot[:, 1] = normed_characteristic_values[:, 9].real
        array_to_plot[:, 2] = normed_characteristic_values[:, -1].real
    else:
        raise ValueError(
            f"Invalid display_type choice; display_type={args['display_type']}"
        )

    return array_to_plot


def plot_char_value_vs_time_with_ref_flow(args: dict, axes: np.ndarray = None):

    if axes is None:
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True)

    eval_times = plot_characteristic_value_vs_time(args, axes=axes[0])

    args["ref_start_time"] = np.min(eval_times)
    args["ref_end_time"] = np.max(eval_times)

    if cfg.MODEL == Models.SHELL_MODEL:
        sh_plot.plot_energy(
            args,
            axes=axes[1],
            plot_args=[],
        )
    elif cfg.MODEL == Models.LORENTZ63:
        l63_plot.plot_energy(args, axes=axes[1])


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()
    compare_plot_arg_parser = a_parsers.ComparisonPlottingArgParser()
    compare_plot_arg_parser.setup_parser()
    args: dict = compare_plot_arg_parser.args

    a_utils.react_on_comparison_arguments(args)

    # Shell model specific
    if cfg.MODEL == Models.SHELL_MODEL:
        # Initiate and update variables and arrays
        sh_utils.update_dependent_params(params)
        sh_utils.set_params(params, parameter="sdim", value=args["sdim"])
        sh_utils.update_arrays(params)

    g_ui.confirm_run_setup(args)
    plt_config.adjust_default_fig_axes_settings(args)

    if "pert_comp_compare" in args["plot_type"]:
        plt_pert_components(args)
    elif "bv_lyap_compare" in args["plot_type"]:
        plt_BV_LYAP_vector_comparison(args)
    elif "error_norm_compare" in args["plot_type"]:
        plot_error_norm_comparison(args)
    elif "rmse_spread_compare" in args["plot_type"]:
        plot_RMSE_and_spread_comparison(args)
    elif "exp_growth_rate_compare" in args["plot_type"]:
        plot_exp_growth_rate_comparison(args)
    elif "char_values_compare" in args["plot_type"]:
        plot_char_value_vs_time_with_ref_flow(args)
    elif "vec_compare_to_lv" in args["plot_type"]:
        plt_vec_compared_to_lv(args)
    elif "tlm_validity" in args["plot_type"]:
        plot_tlm_validity(args)
    else:
        raise ValueError("No valid plot type given as input argument")

    g_plt_utils.save_or_show_plot(args, tight_layout_rect=[0, 0, 0.9, 1])
