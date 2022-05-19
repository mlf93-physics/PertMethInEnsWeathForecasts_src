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
    return_all : bool
        Whether to return both high and low pred regimes

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
        regime_index = int(header[args["regime_start"]])
        if regime_index == 0:  # high pred
            index1 = np.s_[:]
            index2 = index1
        else:  # low pred
            index1 = np.s_[:-1]
            index2 = np.s_[1:]

        start_times = list(regime_start_time_data[index1, regime_index])
        # start_times = list(
        #     abs(
        #         regime_start_time_data[index1, regime_index]
        #         + regime_start_time_data[
        #             index2,
        #             abs(regime_index - 1),
        #         ]
        #     )
        #     / 2
        # )

    return start_times, regime_start_times_shape[0], header


def map_time_to_regime(
    start_times: np.ndarray, regime_start_times: np.ndarray
) -> np.ndarray:
    """Map an array of time values to a regime, i.e. it is determined whether
    a time value belongs to a high or low pred regime

    Parameters
    ----------
    start_times : np.ndarray
        The time values to map
    regime_start_times : np.ndarray
        The start times of the different regimes

    Returns
    -------
    np.ndarray
        An array that tells if a time value belongs to a high (False) or low (True)
        pred regime.

    Raises
    ------
    ValueError
        Raised if the minimum/maximum start_time is smaller/larger than
        the minimum/maximum regime_start_time
    """

    if np.min(start_times) < np.min(regime_start_times) or np.max(start_times) > np.max(
        regime_start_times
    ):
        raise ValueError(
            "The minimum/maximum start_time is smaller/larger than the minimum/maximum regime_start_time"
        )

    flat_regime_start_times: np.ndarray = regime_start_times.ravel()
    index_in_regime_array: np.ndarray = np.searchsorted(
        flat_regime_start_times, start_times
    )

    # Whether a start_time belongs to a high or low pred regime is determined by
    # if the index_in_regime_array entries are even (high pred) or odd (low
    # pred).
    regimes: np.ndarray = (index_in_regime_array % 2 == 0).astype(np.int16)

    return regimes
