import numpy as np
import matplotlib.pyplot as plt
import shell_model_experiments.params as sh_params
import lorentz63_experiments.params.params as l63_params
import general.utils.importing.import_data_funcs as g_import
import general.utils.plot_utils as g_plt_utils
from general.params.experiment_licences import Experiments as EXP
from general.params.model_licences import Models
from config import MODEL, LICENCE

# Get parameters for model
if MODEL == Models.SHELL_MODEL:
    params = sh_params
elif MODEL == Models.LORENTZ63:
    params = l63_params


def analyse_error_norm_vs_time(u_stores, args=None):

    if len(u_stores) == 0:
        raise IndexError("Not enough u_store arrays to compare.")

    if args["combinations"]:
        combinations = [
            [j, i] for j in range(len(u_stores)) for i in range(j + 1) if j != i
        ]
        error_norm_vs_time = np.zeros((u_stores[0].shape[0], len(combinations)))

        for enum, indices in enumerate(combinations):
            error_norm_vs_time[:, enum] = np.linalg.norm(
                u_stores[indices[0]] - u_stores[indices[1]], axis=1
            ).real
    else:
        error_norm_vs_time = np.zeros((u_stores[0].shape[0], len(u_stores)))

        for i in range(len(u_stores)):
            error_norm_vs_time[:, i] = np.linalg.norm(u_stores[i], axis=1).real

    error_norm_mean_vs_time = np.mean(error_norm_vs_time, axis=1)

    return error_norm_vs_time, error_norm_mean_vs_time


def analyse_error_spread_vs_time_norm_of_mean(u_stores, args=None):
    """Calculates the spread of the error (u' - u) using the 'norm of the mean.'

    Formula: ||\sqrt{<((u' - u) - <u' - u>)²>}||

    """

    u_mean = np.mean(np.array(u_stores), axis=0)

    error_spread = np.array([(u_stores[i] - u_mean) ** 2 for i in range(len(u_stores))])

    error_spread = np.sqrt(np.mean(error_spread, axis=0))
    error_spread = np.linalg.norm(error_spread, axis=1)

    return error_spread


def analyse_error_spread_vs_time_mean_of_norm(u_stores, args=None):
    """Calculates the spread of the error (u' - u) using the 'mean of the norm'

    Formula: \sqrt{<(||u' - u|| - <||u' - u||>)²>}

    """
    u_mean_norm = np.mean(np.linalg.norm(np.array(u_stores), axis=2).real, axis=0)

    error_spread = np.array(
        [
            (np.linalg.norm(u_stores[i], axis=1).real - u_mean_norm) ** 2
            for i in range(len(u_stores))
        ]
    )

    error_spread = np.sqrt(np.mean(error_spread, axis=0))

    return error_spread


def plot_error_norm_vs_time(args=None, normalize_start_time=True, axes=None):

    try:
        exp_setup = g_import.import_exp_info_file(args)
    except ImportError:
        print(
            "The .json config file was not found, so this plot doesnt work "
            + "if the file is needed"
        )

    (
        u_stores,
        perturb_time_pos_list,
        perturb_time_pos_list_legend,
        header_dict,
        u_ref_stores,
    ) = g_import.import_perturbation_velocities(args, search_pattern="*perturb*.csv")

    num_perturbations = len(perturb_time_pos_list)

    error_norm_vs_time, error_norm_mean_vs_time = analyse_error_norm_vs_time(
        u_stores, args=args
    )
    if args["plot_mode"] == "detailed":
        error_spread_vs_time = analyse_error_spread_vs_time_mean_of_norm(
            u_stores, args=args
        )

    time_array = np.linspace(
        0,
        header_dict["time_to_run"],
        int(header_dict["time_to_run"] * params.tts) + args["endpoint"] * 1,
        dtype=np.float64,
        endpoint=args["endpoint"],
    )
    if not normalize_start_time:
        time_array = np.repeat(
            np.reshape(time_array, (time_array.size, 1)), num_perturbations, axis=1
        )

        time_array += np.reshape(
            np.array(perturb_time_pos_list) * params.stt, (1, num_perturbations)
        )

    # Pick out specified runs
    if args["specific_files"] is not None:
        perturb_time_pos_list_legend = [
            perturb_time_pos_list_legend[i] for i in args["specific_files"]
        ]
        error_norm_vs_time = error_norm_vs_time[:, args["specific_files"]]

    if args["plot_mode"] == "detailed":
        perturb_time_pos_list_legend = np.append(
            perturb_time_pos_list_legend, ["Mean error norm", "Std of error"]
        )

    # Prepare axes
    if axes is None:
        axes = plt.gca()

    # Get non-repeating colorcycle
    if LICENCE == EXP.BREEDING_VECTORS or LICENCE == EXP.LYAPUNOV_VECTORS:
        n_colors = exp_setup["n_vectors"]
    else:
        n_colors = num_perturbations

    cmap_list = g_plt_utils.get_non_repeating_colors(n_colors=n_colors)
    axes.set_prop_cycle("color", cmap_list)

    axes.plot(time_array, error_norm_vs_time)  # , 'k', linewidth=1)

    if args["plot_mode"] == "detailed":
        # Plot perturbation error norms
        axes.plot(time_array, error_norm_mean_vs_time, "k")  # , 'k', linewidth=1)
        # Plot mean perturbation error norm
        axes.plot(time_array, error_spread_vs_time, "k--")  # , 'k', linewidth=1)
        # Plot std of perturbation errors

    axes.set_xlabel("Time")
    axes.set_ylabel("Error")
    axes.set_yscale("log")

    if not LICENCE == EXP.BREEDING_VECTORS or LICENCE == EXP.LYAPUNOV_VECTORS:
        axes.legend(perturb_time_pos_list_legend)

    if args["xlim"] is not None:
        axes.set_xlim(args["xlim"][0], args["xlim"][1])
    if args["ylim"] is not None:
        axes.set_ylim(args["ylim"][0], args["ylim"][1])

    if MODEL == Models.SHELL_MODEL:
        axes.set_title(
            f'Error vs time; f={header_dict["forcing"]}'
            + f', $n_f$={int(header_dict["n_f"])}, $\\nu$={header_dict["ny"]:.2e}'
            + f', time={header_dict["time_to_run"]} | Experiment: {args["exp_folder"]};'
            + f'Files: {args["file_offset"]}-{args["file_offset"] + args["n_files"]}'
        )
    elif MODEL == Models.LORENTZ63:
        axes.set_title(
            f'Error vs time; sigma={header_dict["sigma"]}'
            + f', $b$={header_dict["b_const"]:.2e}, r={header_dict["r_const"]}'
            + f', time={header_dict["time_to_run"]} | Experiment: {args["exp_folder"]};'
            + f'Files: {args["file_offset"]}-{args["file_offset"] + args["n_files"]}'
        )

    # plt.savefig(f'../figures/week6/error_eigen_spectrogram/error_norm_ny{header_dict["ny"]:.2e}_file_{args["file_offset"]}', format='png')

    # print('perturb_time_pos_list', perturb_time_pos_list)

    # Plot energy below
    # ref_file_name = list(Path(args['path']).glob('*.csv'))
    # data_in, ref_header_dict = g_import.import_data(ref_file_name[0],
    #     start_line=int(perturb_time_pos_list[0]*sample_rate/dt) + 1,
    #     max_lines=int((perturb_time_pos_list[-1] - perturb_time_pos_list[0])*sample_rate/dt + header_dict['N_data']))

    # # print('perturb_time_pos_list[0]', perturb_time_pos_list[0])
    # plot_inviscid_quantities(data_in[:, 0] - perturb_time_pos_list[0],
    #     data_in[:, 1:], ref_header_dict, ax=axes[1], args=args,
    #     zero_time_ref=int(perturb_time_pos_list[0]*sample_rate/dt))

    return axes
