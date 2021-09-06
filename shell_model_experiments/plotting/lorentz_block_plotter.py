import sys

sys.path.append("..")
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pyinstrument import Profiler
from shell_model_experiments.params.params import *
from shell_model_experiments.utils.import_data_funcs import (
    import_perturbation_velocities,
)
import shell_model_experiments.analyses.lorentz_block_analysis as lr_analysis

profiler = Profiler()


def plt_lorentz_block_from_full_perturbation_data(args):

    if "perturb_folder" in args:

        parent_pert_folder = args["perturb_folder"]
        # Import forecasts
        args["perturb_folder"] = parent_pert_folder + "/forecasts"
        args["n_files"] = -1

        (
            forecast_pert_u_stores,
            _,
            _,
            forecast_header_dict,
        ) = import_perturbation_velocities(args)

        # Import forecasts
        args["perturb_folder"] = parent_pert_folder + "/analysis_forecasts"
        args["n_files"] = -1

        (
            ana_forecast_pert_u_stores,
            _,
            _,
            _,
        ) = import_perturbation_velocities(args)

        num_ana_forecasts = len(ana_forecast_pert_u_stores)
        num_forecasts = len(forecast_pert_u_stores)
        rmse_array = np.zeros((num_forecasts, num_ana_forecasts), dtype=np.float64)
        day_offset = forecast_header_dict["start_time_offset"]

        for fc in range(num_forecasts):
            for day in range(fc, num_ana_forecasts):
                if day == fc:

                    # NOTE: reference velocities are subtracted on import, so
                    # this is the forecast error directly
                    _error = forecast_pert_u_stores[fc][
                        int((day + 1) * day_offset * sample_rate / dt) + 1, :
                    ]
                    rmse_array[fc, day] = np.sqrt(
                        np.mean((_error * _error.conj()).real)
                    )
                else:
                    _error = (
                        forecast_pert_u_stores[fc][
                            int((day + 1) * day_offset * sample_rate / dt) + 1, :
                        ]
                        - ana_forecast_pert_u_stores[fc][
                            int((day - fc) * day_offset * sample_rate / dt) + 1, :
                        ]
                    )
                    rmse_array[fc, day] = np.sqrt(
                        np.mean((_error * _error.conj()).real)
                    )

        rmse_array[np.where(rmse_array == 0)] = float("nan")
        plt.plot(rmse_array[:num_forecasts, :].T)
        plt.show()


def plt_lorentz_block(args):

    rmse_mean = lr_analysis.lorentz_block_analyser.analysis_executer(args)

    plt.plot(rmse_mean.T)
    plt.show()


if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", nargs="?", default=None, type=str)
    arg_parser.add_argument("--perturb_folder", nargs="?", default=None, type=str)
    arg_parser.add_argument("--n_files", default=-1, type=int)
    arg_parser.add_argument("--experiment", nargs="?", default=None, type=str)

    args = vars(arg_parser.parse_args())

    # Add missing arguments to make util funcs work
    args["specific_ref_records"] = [0]
    args["file_offset"] = 0

    profiler.start()

    plt_lorentz_block(args)

    profiler.stop()
    print(profiler.output_text())
