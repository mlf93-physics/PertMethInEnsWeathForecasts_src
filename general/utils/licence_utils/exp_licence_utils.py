import re
import pathlib as pl
import sys
from general.params.experiment_licences import Experiments, Experiment


def detect_exp_licence() -> Experiment:
    """Detect which licence to use

    Returns
    -------
    Experiment
        The valid experiment licence
    """
    # Get name of root file
    root_file_name = pl.Path(sys.argv[0]).stem

    # Get experiments
    exp = Experiments()

    if "perturbation_runner" == root_file_name:
        licence = exp.NORMAL_PERTURBATION

    elif "tangent_linear" in root_file_name:
        licence = exp.TANGENT_LINEAR

    elif "breed_vector_eof" in root_file_name:
        licence = exp.BREEDING_EOF_VECTORS

    elif "breed_vector" in root_file_name:
        licence = exp.BREEDING_VECTORS

    elif "hyper_diff_perturb_runner" == root_file_name:
        licence = exp.HYPER_DIFFUSIVITY

    elif "lorentz_block_runner" == root_file_name:
        licence = exp.LORENTZ_BLOCK

    elif bool(re.match(r"lyapunov_vector\w+", root_file_name)):
        licence = exp.LYAPUNOV_VECTORS

    elif bool(re.match(r"adj_lyapunov_vector\w+", root_file_name)):
        licence = exp.ADJ_LYAPUNOV_VECTORS

    elif bool(re.match(r"singular_vector\w+", root_file_name)):
        licence = exp.SINGULAR_VECTORS

    elif bool(re.match(r"final_singular_vector\w+", root_file_name)):
        licence = exp.FINAL_SINGULAR_VECTORS

    elif "compare" in root_file_name:
        licence = exp.COMPARISON

    elif "veri" in root_file_name:
        licence = exp.VERIFICATION

    elif "plot_data" in root_file_name:
        licence = exp.NORMAL_PERTURBATION

    else:
        licence = None
        print("\nNo experiment licence matches the current root file. LICENCE=None\n")

    return licence
