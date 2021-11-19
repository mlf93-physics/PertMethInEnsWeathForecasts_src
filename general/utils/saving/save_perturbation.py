import numpy as np
import general.utils.saving.save_lorentz_block_funcs as lb_save
from general.params.experiment_licences import Experiments as EXP
import general.utils.saving.save_data_funcs as g_save
import config as cfg


def save_perturbation_data(
    data_out: np.ndarray,
    perturb_position: int = None,
    perturb_count: int = None,
    args: dict = None,
):
    """Save the perturbation data depending on the LICENCE

    Parameters
    ----------
    data_out : np.ndarray
        The numpy array containing data to save
    perturb_position : int, optional
        The perturbation position as index, by default None
    perturb_count : int, optional
        The count of the current perturbation, by default None
    args : dict, optional
        Run-time arguments, by default None
    """
    if cfg.LICENCE == EXP.LORENTZ_BLOCK:
        lb_save.save_lorentz_block_data(
            data_out,
            prefix=f"lorentz{perturb_count}_",
            perturb_position=perturb_position,
            args=args,
        )
        return
    elif cfg.LICENCE == EXP.NORMAL_PERTURBATION:
        prefix = f"perturb{perturb_count}_"
    elif cfg.LICENCE == EXP.BREEDING_VECTORS:
        prefix = f"breed_perturb{perturb_count}_"
    elif cfg.LICENCE == EXP.LYAPUNOV_VECTORS:
        prefix = f"lyapunov_perturb{perturb_count}_"
    elif cfg.LICENCE == EXP.HYPER_DIFFUSIVITY:
        prefix = f"hyper_perturb{perturb_count}_"
    elif cfg.LICENCE == EXP.VERIFICATION:
        prefix = f"verification_perturb{perturb_count}_"
    else:
        print(f"No saving method present for the current licence ({cfg.LICENCE})")

    g_save.save_data(
        data_out,
        prefix=prefix,
        perturb_position=perturb_position,
        args=args,
    )
