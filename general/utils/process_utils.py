import sys

sys.path.append("..")
import pathlib as pl
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_utils as g_save_utils
import general.runners.perturbation_runner as pt_runner


def run_pert_processes(args: dict, local_exp_setup: dict, processes: list):
    """Run a list of perturbation processes, save experiment info and possibly
    compress the data dir if in ERDA mode.

    Parameters
    ----------
    args : dict
        Local run-time arguments
    local_exp_setup : dict
        Local experiment setup
    processes : list
        The processes to run
    """

    if len(processes) > 0:
        pt_runner.main_run(
            processes,
            args=args,
            n_units=args["n_units"],
        )
        # Save exp setup to exp folder
        g_save.save_exp_info(local_exp_setup, args)

        if args["erda_run"]:
            path = pl.Path(args["datapath"], local_exp_setup["folder_name"])
            g_save_utils.compress_dir(path)
    else:
        print("No processes to run - check if blocks already exists")
