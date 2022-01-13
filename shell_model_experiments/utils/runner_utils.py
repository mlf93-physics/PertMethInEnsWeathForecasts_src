import colorama as col
import pathlib as pl
from general.utils.module_import.type_import import *
import numpy as np
import general.utils.importing.import_data_funcs as g_import
import general.utils.exceptions as g_exceptions
from libs.libutils import file_utils as lib_file_utils


def get_regime_start_times(args: dict, return_all=False):
    """Get the start times for a given regime (high pred or low pred)

    Parameters
    ----------
    args : dict
        Run-time arguments

    Raises
    ------
    ImportError
        Raised if multiple regime_start_time analysis files found
    g_exceptions.InvalidRuntimeArgument
        Raised if the number of requested profiles is larger than the available
        regime start time

    Returns
    -------
    start_times : list or np.ndarray
        The regime start times
    """

    # Find analysis file
    regime_start_time_path: List[pl.Path] = lib_file_utils.get_files_in_path(
        pl.Path(args["datapath"], "analysis_data"),
        search_pattern="regime_start_times*.csv",
    )
    if len(regime_start_time_path) > 1:
        raise ImportError(
            f"Multiple regime_start_time analysis files found at path {regime_start_time_path}"
        )

    # Import data
    regime_start_time_data, header = g_import.import_data(
        regime_start_time_path[0], dtype=np.float64
    )

    regime_start_times_shape = regime_start_time_data.shape
    if args["n_profiles"] > regime_start_times_shape[0]:
        raise g_exceptions.InvalidRuntimeArgument(
            "Number of requested profiles is larger than the available regime start time; "
            + f"{args['n_profiles']}>{regime_start_times_shape[0]}"
        )

    if return_all:
        print(
            f"\n{col.Fore.RED}In get_regime_start_times: Both high and low pred start time requested to be returned\n{col.Fore.RESET}"
        )
        start_times = regime_start_time_data
    else:
        start_times = list(regime_start_time_data[:, int(header[args["regime_start"]])])

    return start_times, regime_start_times_shape[0], header
