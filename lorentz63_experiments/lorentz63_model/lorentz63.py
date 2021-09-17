import sys

sys.path.append("..")
import math
import argparse
from pyinstrument import Profiler
import numpy as np
from numba import njit, types
import lorentz63_experiments.lorentz63_model.runge_kutta4 as rk4
from lorentz63_experiments.params.params import *
import lorentz63_experiments.utils.util_funcs as ut_funcs
import general.utils.save_data_funcs as g_save
from config import NUMBA_CACHE


profiler = Profiler()


@njit(
    (
        types.Array(types.float64, 1, "C", readonly=False),
        types.Array(types.float64, 1, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
        types.int64,
    ),
    cache=NUMBA_CACHE,
)
def run_model(x_old, du_array, derivMatrix, data_out, Nt_local):
    """Execute the integration of the sabra shell model.

    Parameters
    ----------
    x_old : ndarray
        The initial lorentz positions
    du_array : ndarray
        A helper array used to store the current derivative of the lorentz
        positions.
    data_out : ndarray
        An array to store samples of the integrated lorentz_positions.

    """

    sample_number = 0
    # Perform calculations
    for i in range(Nt_local):
        # Save samples for plotting
        if i % int(1 / sample_rate) == 0:
            data_out[sample_number, 0] = dt * i
            data_out[sample_number, 1:] = x_old
            sample_number += 1

        # Update x_old
        x_old = rk4.runge_kutta4_vec(
            y0=x_old, h=dt, dx=du_array, derivMatrix=derivMatrix
        )

    return x_old


def main(args=None):

    # Define x_old
    x_old = np.array([1, 1, 1], dtype=np.float64)

    derivMatrix = ut_funcs.setup_deriv_matrix(args)

    # Get number of records
    args["n_records"] = math.ceil(
        (args["Nt"] - args["burn_in_time"] / dt) / int(args["record_max_time"] / dt)
    )

    profiler.start()
    print(
        f'\nRunning Lorentz63 model for {args["Nt"]*dt:.2f}s with a burn-in time'
        + f' of {args["burn_in_time"]:.2f}s, i.e. {args["n_records"]:d} records '
        + f'are saved to disk each with {args["record_max_time"]:.1f}s data\n'
    )

    # Burn in the model for the desired burn in time
    data_out = np.zeros((int(args["burn_in_time"] * tts), sdim + 1), dtype=np.float64)
    print(f'Running burn-in phase of {args["burn_in_time"]}s\n')
    x_old = run_model(
        x_old, du_array, derivMatrix, data_out, int(args["burn_in_time"] / dt)
    )

    for ir in range(args["n_records"]):
        # Calculate data out size
        if ir == (args["n_records"] - 1):
            if (args["Nt"] - args["burn_in_time"] / dt) % int(
                args["record_max_time"] / dt
            ) > 0:
                out_array_size = int(
                    (
                        args["Nt"]
                        - args["burn_in_time"] / dt % int(args["record_max_time"] / dt)
                    )
                    * sample_rate
                )
            else:
                out_array_size = int(args["record_max_time"] * tts)
        else:
            out_array_size = int(args["record_max_time"] * tts)

        data_out = np.zeros((out_array_size, sdim + 1), dtype=np.float64)

        # Run model
        print(f'running record {ir + 1}/{args["n_records"]}')
        x_old = run_model(
            x_old,
            du_array,
            derivMatrix,
            data_out,
            int(out_array_size / sample_rate),
        )

        # Add record_id to datafile header
        args["record_id"] = ir

        if args["save_data"]:
            print(f"saving record\n")
            g_save.save_data(data_out, args=args)

        # pl_data.plot_attractor(data_out)

    profiler.stop()
    print(profiler.output_text())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--burn_in_time", default=0.0, type=float)
    arg_parser.add_argument("--save_folder", nargs="?", default="data", type=str)
    arg_parser.add_argument("--record_max_time", default=1000, type=float)
    arg_parser.add_argument("--save_data", action="store_false")
    arg_parser.add_argument("--time_to_run", type=float, required=True)
    arg_parser.add_argument("--sigma", default=10, type=float)
    arg_parser.add_argument("--r_const", default=28, type=float)
    arg_parser.add_argument("--b_const", default=8 / 3, type=float)

    args = vars(arg_parser.parse_args())

    # Add/edit arguments
    args["ref_run"] = True
    args["Nt"] = int(args["time_to_run"] / dt)

    main(args)
