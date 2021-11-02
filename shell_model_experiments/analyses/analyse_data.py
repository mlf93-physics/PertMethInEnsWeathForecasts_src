import sys

sys.path.append("..")
import numpy as np
from general.utils.module_import.type_import import *
import general.utils.importing.import_data_funcs as g_import
import general.utils.saving.save_data_funcs as g_save
import general.utils.argument_parsers as a_parsers
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
import general.utils.exceptions as g_exceptions
import config as cfg

cfg.GLOBAL_PARAMS.ref_run = False


def analyse_mean_energy_spectrum(args: dict) -> Tuple[np.ndarray, dict]:

    _, u_data, header_dict = g_import.import_ref_data(args=args)

    g_utils.determine_params_from_header_dict(header_dict, args)

    mean_energy = np.mean(
        (u_data * np.conj(u_data)).real,
        axis=0,
    )

    return mean_energy, header_dict


def analyse_mean_energy_spectra(args: dict):

    if args["datapaths"] is None:
        raise g_exceptions.InvalidRuntimeArgument(
            "Argument not set", argument="datapaths"
        )

    for i, path in enumerate(args["datapaths"]):
        args["datapath"] = path

        mean_energy, _ = analyse_mean_energy_spectrum(args)

        mean_energy = np.reshape(mean_energy, (1, mean_energy.size))

        g_save.save_data(mean_energy, prefix="mean_energy", args=args)


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    stand_plot_arg_parser = a_parsers.StandardPlottingArgParser()
    stand_plot_arg_parser.setup_parser()

    args = stand_plot_arg_parser.args

    g_ui.confirm_run_setup(args)

    analyse_mean_energy_spectra(args)
