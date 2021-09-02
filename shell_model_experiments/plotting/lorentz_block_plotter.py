import sys

sys.path.append("..")
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from shell_model_experiments.params.params import *
from shell_model_experiments.utils.import_data_funcs import (
    import_data,
    import_header,
    import_ref_data,
    import_perturbation_velocities,
    import_start_u_profiles,
)


def plt_lorentz_block(args):

    time, u_data, header_dict = import_ref_data(args=args)
    if "perturb_folder" in args:

        # forecast_file_names = list(
        #     Path(args["path"], args["perturb_folder"], "forecasts").glob("*.csv")
        # )
        # analysis_forecast_file_names = list(
        #     Path(args["path"], args["perturb_folder"], "analysis_forecasts").glob(
        #         "*.csv"
        #     )
        # )

        # Import forecasts
        args["perturb_folder"] += "/forecasts"
        args["n_files"] = 1

        (
            forecast_u_stores,
            forecast_time_pos_list,
            forecast_time_pos_list_legend,
            forecast_header_dict,
        ) = import_perturbation_velocities(args)

        print(forecast_u_stores)


if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", nargs="?", default=None, type=str)
    arg_parser.add_argument("--perturb_folder", nargs="?", default=None, type=str)
    arg_parser.add_argument("--n_files", default=-1, type=int)

    args = vars(arg_parser.parse_args())

    # Add missing arguments to make util funcs work
    args["specific_ref_records"] = [0]
    args["file_offset"] = 0

    plt_lorentz_block(args)
