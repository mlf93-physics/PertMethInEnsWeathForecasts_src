import sys

sys.path.append("..")
import math
from pyinstrument import Profiler
import numpy as np
from numba import njit, types
import lorentz63_experiments.lorentz63_model.runge_kutta4 as rk4
from lorentz63_experiments.params.params import *
import lorentz63_experiments.utils.util_funcs as ut_funcs
import general.utils.saving.save_data_funcs as g_save
import general.utils.argument_parsers as a_parsers
import config as cfg

profiler = Profiler()

# Set global params
cfg.GLOBAL_PARAMS.record_max_time = 3000


@njit(
    (
        types.Array(types.float64, 1, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
        types.Array(types.float64, 2, "C", readonly=False),
        types.int64,
    ),
    cache=cfg.NUMBA_CACHE,
)
def run_model(u_old, lorentz_matrix, data_out, Nt_local):
    """Execute the integration of the Lorentz-63 model.

    Parameters
    ----------
    u_old : ndarray
        The initial lorentz velocities
    du_array : ndarray
        A helper array used to store the current derivative of the lorentz
        velocities.
    data_out : ndarray
        An array to store samples of the integrated lorentz_velocities.

    """

    sample_number = 0
    # Perform calculations
    for i in range(Nt_local):
        # Save samples for plotting
        if i % int(1 / sample_rate) == 0:
            data_out[sample_number, 0] = dt * i
            data_out[sample_number, 1:] = u_old
            sample_number += 1

        # Update u_old
        u_old = rk4.runge_kutta4(y0=u_old, lorentz_matrix=lorentz_matrix, h=dt)

    return u_old


def main(args=None):

    # Define u_old
    u_old = np.array(
        [1, 1, 1],
        dtype=np.float64,
    )

    lorentz_matrix = ut_funcs.setup_lorentz_matrix(args)

    # Get number of records
    args["n_records"] = math.ceil(
        args["Nt"] / int(cfg.GLOBAL_PARAMS.record_max_time / dt)
    )

    # Write ref info file
    g_save.save_reference_info(args)

    profiler.start()
    print(
        f'\nRunning Lorentz63 model for {args["Nt"]*dt:.2f}s with a burn-in time'
        + f' of {args["burn_in_time"]:.2f}s, i.e. {args["n_records"]:d} records '
        + f"are saved to disk each with {cfg.GLOBAL_PARAMS.record_max_time:.1f}s data\n"
    )

    # Burn in the model for the desired burn in time
    data_out = np.zeros((int(args["burn_in_time"] * tts), sdim + 1), dtype=np.float64)
    print(f'Running burn-in phase of {args["burn_in_time"]}s\n')
    u_old = run_model(u_old, lorentz_matrix, data_out, int(args["burn_in_time"] / dt))

    for ir in range(args["n_records"]):
        # Calculate data out size
        if ir == (args["n_records"] - 1):
            if args["Nt"] % int(cfg.GLOBAL_PARAMS.record_max_time / dt) > 0:
                out_array_size = int(
                    (args["Nt"] % int(cfg.GLOBAL_PARAMS.record_max_time / dt))
                    * sample_rate
                )
            else:
                out_array_size = int(cfg.GLOBAL_PARAMS.record_max_time * tts)
        else:
            out_array_size = int(cfg.GLOBAL_PARAMS.record_max_time * tts)

        data_out = np.zeros((out_array_size, sdim + 1), dtype=np.float64)

        # Run model
        print(f'running record {ir + 1}/{args["n_records"]}')
        u_old = run_model(
            u_old,
            lorentz_matrix,
            data_out,
            int(out_array_size / sample_rate),
        )

        # Add record_id to datafile header
        args["record_id"] = ir

        if not args["skip_save_data"]:
            print(f"saving record\n")
            g_save.save_data(data_out, args=args)

    profiler.stop()
    print(profiler.output_text(color=True))


if __name__ == "__main__":
    # Get arguments
    stand_arg_setup = a_parsers.StandardRunnerArgSetup()
    stand_arg_setup.setup_parser()
    args = stand_arg_setup.args

    # initiate_sdim_arrays(args["sdim"])

    # Add/edit arguments
    args["Nt"] = int(args["time_to_run"] / dt)

    main(args)
