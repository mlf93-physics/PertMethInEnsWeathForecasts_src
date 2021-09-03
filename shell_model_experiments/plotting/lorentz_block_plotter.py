import sys

sys.path.append("..")
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from shell_model_experiments.params.params import *
from shell_model_experiments.utils.import_data_funcs import (
    import_perturbation_velocities,
    import_lorentz_block_perturbations,
)


def plt_lorentz_block(args):

    if "perturb_folder" in args:

        parent_pert_folder = args["perturb_folder"]
        # Import forecasts
        args["perturb_folder"] = parent_pert_folder + "/forecasts"
        args["n_files"] = -1

        (
            forecast_pert_u_stores,
            forecast_time_pos_list,
            forecast_time_pos_list_legend,
            forecast_header_dict,
        ) = import_perturbation_velocities(args)

        # Import forecasts
        args["perturb_folder"] = parent_pert_folder + "/analysis_forecasts"
        args["n_files"] = -1

        (
            ana_forecast_pert_u_stores,
            ana_forecast_time_pos_list,
            ana_forecast_time_pos_list_legend,
            ana_forecast_header_dict,
        ) = import_perturbation_velocities(args)

        num_ana_forecasts = len(ana_forecast_pert_u_stores)
        num_forecasts = len(forecast_pert_u_stores)
        rmse_array = np.zeros((num_forecasts, num_ana_forecasts), dtype=np.float64)
        day_offset = 0.01

        for fc in range(num_forecasts):
            for day in range(fc, num_ana_forecasts):
                print("fc", fc, "day", day)
                if day == fc:
                    # print("fc ref", fc)
                    # print(
                    #     "ref error indices:",
                    #     int((day + 1) * day_offset * sample_rate / dt) + 1,
                    #     "(fc)",
                    #     int(ana_forecast_time_pos_list[fc]),
                    #     "(ref)",
                    # )

                    # NOTE: reference velocities are subtracted on import, so
                    # this is the forecast error directly
                    _error = forecast_pert_u_stores[fc][
                        int((day + 1) * day_offset * sample_rate / dt) + 1, :
                    ]
                    rmse_array[fc, day] = np.sqrt(
                        np.mean((_error * _error.conj()).real)
                    )
                else:
                    # print(
                    #     "fc",
                    #     forecast_pert_u_stores[fc][
                    #         int((day + 1) * day_offset * sample_rate / dt) + 1, :
                    #     ],
                    #     "ana",
                    #     ana_forecast_pert_u_stores[fc][
                    #         int((day - fc) * day_offset * sample_rate / dt) + 1, :
                    #     ],
                    # )

                    _error = (
                        forecast_pert_u_stores[fc][
                            int((day + 1) * day_offset * sample_rate / dt) + 1, :
                        ]
                        - ana_forecast_pert_u_stores[fc][
                            int((day - fc) * day_offset * sample_rate / dt) + 1, :
                        ]
                    )
                    print(
                        "fc old",
                        forecast_pert_u_stores[fc][
                            int((day + 1) * day_offset * sample_rate / dt) + 1, :
                        ],
                    )
                    input()
                    rmse_array[fc, day] = np.sqrt(
                        np.mean((_error * _error.conj()).real)
                    )

        rmse_array[np.where(rmse_array == 0)] = float("nan")
        print("rmse_array", rmse_array)
        plt.plot(rmse_array[:num_forecasts, :].T)
        plt.show()


def plt_lorentz_block2(args):

    if "perturb_folder" in args:

        parent_pert_folder = args["perturb_folder"]
        # Import forecasts
        args["perturb_folder"] = parent_pert_folder + "/forecasts"
        args["n_files"] = -1

        (
            forecast_pert_u_stores,
            forecast_time_pos_list,
            forecast_time_pos_list_legend,
            forecast_header_dict,
        ) = import_lorentz_block_perturbations(args)

        # Import analyses forecasts
        args["perturb_folder"] = parent_pert_folder + "/analysis_forecasts"
        args["n_files"] = -1

        (
            ana_forecast_pert_u_stores,
            ana_forecast_time_pos_list,
            ana_forecast_time_pos_list_legend,
            ana_forecast_header_dict,
        ) = import_lorentz_block_perturbations(args)

        num_ana_forecasts = len(ana_forecast_pert_u_stores)
        num_forecasts = len(forecast_pert_u_stores)
        rmse_array = np.zeros((num_forecasts, num_ana_forecasts), dtype=np.float64)

        for fc in range(num_forecasts):
            for day in range(fc, num_ana_forecasts):
                print("fc", fc, "day", day)
                if day == fc:
                    # print("fc ref", fc)

                    # NOTE: reference velocities are subtracted on import, so
                    # this is the forecast error directly
                    _error = forecast_pert_u_stores[fc][day, :]
                    rmse_array[fc, day] = np.sqrt(
                        np.mean((_error * _error.conj()).real)
                    )

                else:
                    # NOTE: to ana_forecast_pert_u_stores[fc][(day - fc - 1), :]:
                    # (day - 1): -1 since analysis and forecast data is
                    # offset by one index in data.
                    # print(
                    #     "forecast_pert_u_stores[fc]", forecast_pert_u_stores[fc].shape
                    # )
                    # print("day", day)
                    # print(
                    #     "fc",
                    #     forecast_pert_u_stores[fc][day, :],
                    #     "ana",
                    #     ana_forecast_pert_u_stores[fc][(day - 1), :],
                    # )
                    _error = (
                        forecast_pert_u_stores[fc][day, :]
                        - ana_forecast_pert_u_stores[fc][(day - fc - 1), :]
                    )
                    print("fc new", forecast_pert_u_stores[fc][day, :])
                    input()
                    rmse_array[fc, day] = np.sqrt(
                        np.mean((_error * _error.conj()).real)
                    )

        rmse_array[np.where(rmse_array == 0)] = float("nan")
        print("rmse_array", rmse_array)
        plt.plot(rmse_array[:num_forecasts, :].T)
        plt.show()


if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", nargs="?", default=None, type=str)
    arg_parser.add_argument("--perturb_folder", nargs="?", default=None, type=str)
    arg_parser.add_argument("--n_files", default=-1, type=int)
    arg_parser.add_argument("--plotter", default="old", type=str)

    args = vars(arg_parser.parse_args())

    # Add missing arguments to make util funcs work
    args["specific_ref_records"] = [0]
    args["file_offset"] = 0

    if args["plotter"] == "old":
        plt_lorentz_block(args)
    elif args["plotter"] == "new":
        plt_lorentz_block2(args)
