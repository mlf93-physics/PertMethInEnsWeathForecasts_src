import sys

sys.path.append("..")
import multiprocessing
import pathlib as pl
import general.utils.saving.save_data_funcs as g_save
import general.utils.saving.save_utils as g_save_utils
from general.utils.module_import.type_import import *


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
        main_run(
            processes,
        )
        # Save exp setup to exp folder
        g_save.save_exp_info(local_exp_setup, args)

        if args["erda_run"]:
            path = pl.Path(args["datapath"], local_exp_setup["folder_name"])
            g_save_utils.compress_dir(path)
    else:
        print("No processes to run - check if blocks already exists")


def main_run(processes: List[multiprocessing.Process]):
    """Run a list of processes in parallel. The processes are distributed according
    to the number of cpu cores

    Parameters
    ----------
    processes : list
        List of processes to run
    """

    cpu_count = multiprocessing.cpu_count()
    num_processes = len(processes)

    for j in range(num_processes // cpu_count):

        for i in range(cpu_count):
            count = j * cpu_count + i
            processes[count].start()

        for i in range(cpu_count):
            count = j * cpu_count + i
            processes[count].join()
            processes[count].close()

    for i in range(num_processes % cpu_count):

        count = (num_processes // cpu_count) * cpu_count + i
        processes[count].start()

    for i in range(num_processes % cpu_count):
        count = (num_processes // cpu_count) * cpu_count + i
        processes[count].join()
        processes[count].close()
