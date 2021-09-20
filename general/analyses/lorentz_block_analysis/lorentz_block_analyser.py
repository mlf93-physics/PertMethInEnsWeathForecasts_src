import sys

sys.path.append("..")
import re
import argparse
import pathlib as pl
import numpy as np
from shell_model_experiments.utils.import_data_funcs import (
    import_lorentz_block_perturbations,
)


def calculate_rmse_of_block(args):

    parent_pert_folder = args["perturb_folder"]
    # Import forecasts
    args["perturb_folder"] = parent_pert_folder + "/forecasts"
    args["n_files"] = np.inf

    (
        forecast_pert_u_stores,
        forecast_ref_u_stores,
        _,
        _,
        _,
    ) = import_lorentz_block_perturbations(args, rel_ref=False)

    # Import analyses forecasts
    args["perturb_folder"] = parent_pert_folder + "/analysis_forecasts"
    args["n_files"] = np.inf

    (
        ana_forecast_pert_u_stores,
        ana_forecast_ref_u_stores,
        _,
        _,
        ana_forecast_pert_header_dicts,
    ) = import_lorentz_block_perturbations(args, rel_ref=False)

    num_ana_forecasts = len(ana_forecast_pert_u_stores)
    num_forecasts = len(forecast_pert_u_stores)
    rmse_array = np.zeros((num_forecasts, num_ana_forecasts), dtype=np.float64)

    for fc in range(num_forecasts):
        for day in range(fc, num_ana_forecasts):
            if day == fc:

                # NOTE: reference velocities are subtracted on import, so
                # this is the forecast error directly
                _error = forecast_pert_u_stores[fc][day, :]
                _ref = forecast_ref_u_stores[fc][day, :]
                rmse_array[fc, day] = np.sqrt(
                    (
                        np.sum((_error * _error.conj()).real)
                        - np.sum((_ref * _ref.conj()).real)
                    )
                    ** 2
                )

            else:
                # NOTE: to ana_forecast_pert_u_stores[fc][(day - fc - 1), :]:
                # (day - 1): -1 since analysis and forecast data is
                # offset by one index in data.
                _error_fc = forecast_pert_u_stores[fc][day, :]
                _error_ana = ana_forecast_pert_u_stores[fc][(day - fc - 1), :]

                rmse_array[fc, day] = np.sqrt(
                    (
                        np.sum((_error_fc * _error_fc.conj()).real)
                        - np.sum((_error_ana * _error_ana.conj()).real)
                    )
                    ** 2
                )

    # Remove lower triangel before plotting
    dummy_tril_matrix = np.tril(np.ones(rmse_array.shape, dtype=np.int8), k=-1)
    rmse_array[np.where(dummy_tril_matrix)] = float("nan")

    return rmse_array, ana_forecast_pert_header_dicts[0]


def get_block_dirs(args):

    experiment_dir = pl.Path(args["path"], args["experiment"])
    block_dirs = list(experiment_dir.glob("*"))

    # Filter out anything else than directories
    block_dirs = [
        block_dirs[i] for i in range(len(block_dirs)) if block_dirs[i].is_dir()
    ]

    # Sort dirs
    block_dirs = [block_dirs[i] for i in np.argsort(block_dirs)]

    # Adjust number of blocks
    if args["num_blocks"] < np.inf and args["num_blocks"] > 0:
        block_dirs = block_dirs[: args["num_blocks"]]

    return block_dirs


def analysis_executer(args):

    block_dirs = get_block_dirs(args)

    rmse = []
    header_dicts = []

    for block in block_dirs:
        args["perturb_folder"] = str(pl.Path(block.parents[0].name, block.name))

        # Only import selected blocks if specified
        if args["specific_blocks"] is not None:
            block_numbers = re.findall(r"\d+", block.name)

            if len(block_numbers) > 1:
                raise ValueError(
                    "Multiple numbers in block name; please only put one ID"
                    + " number in block dir name"
                )

            block_number = int(block_numbers[0])

            if not block_number in args["specific_blocks"]:
                continue

        temp_rmse, ana_forecast_header_dict = calculate_rmse_of_block(args)

        # Append data and header dict
        rmse.append(temp_rmse)
        header_dicts.append(ana_forecast_header_dict)

    if args["specific_blocks"] is None:
        num_imported_blocks = len(block_dirs)
    else:
        num_imported_blocks = len(args["specific_blocks"])

    # Sort rmse and header_dicts lists
    perturb_pos = [header_dicts[i]["perturb_pos"] for i in range(num_imported_blocks)]
    sort_index = np.argsort(perturb_pos)
    header_dicts = [header_dicts[i] for i in sort_index]
    rmse = [rmse[i] for i in sort_index]

    rmse = np.stack(rmse, axis=2)

    if args["average"]:
        rmse = np.log(rmse)
        rmse = np.mean(rmse, axis=2)

    return rmse, header_dicts


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", nargs="?", default=None, type=str)
    arg_parser.add_argument("--experiment", nargs="?", default=None, type=str)
    args = vars(arg_parser.parse_args())

    # Add missing arguments to make util funcs work
    args["specific_ref_records"] = [0]
    args["file_offset"] = 0

    analysis_executer(args)
