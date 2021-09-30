import general.utils.saving.save_lorentz_block_funcs as lb_save
from general.params.experiment_licences import Experiments as EXP
import general.utils.saving.save_data_funcs as g_save
from config import LICENCE


def save_perturbation_data(
    data_out, perturb_position=None, perturb_count=None, args=None
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
    if LICENCE == EXP.NORMAL_PERTURBATION:
        g_save.save_data(
            data_out,
            prefix=f"perturb{perturb_count}_",
            perturb_position=perturb_position,
            args=args,
        )
    elif LICENCE == EXP.LORENTZ_BLOCK:
        lb_save.save_lorentz_block_data(
            data_out,
            prefix=f"lorentz{perturb_count}_",
            perturb_position=perturb_position,
            args=args,
        )
    elif LICENCE == EXP.BREEDING_VECTORS:
        g_save.save_data(
            data_out,
            prefix=f"breed_perturb{perturb_count}_",
            perturb_position=perturb_position,
            args=args,
        )
