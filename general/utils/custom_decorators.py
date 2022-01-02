import colorama as col
from shell_model_experiments.params.params import ParamsStructType
from shell_model_experiments.params.params import PAR as PAR_SH
import shell_model_experiments.utils.util_funcs as sh_utils
from general.params.model_licences import Models
import config as cfg


def check_dimension(header_func: callable) -> callable:
    """Check dimension in header_dict and update parameter struct if mismatch
    detected.

    Parameters
    ----------
    header_func : callable
        The wrapped header function

    Returns
    -------
    callable
        The wrapper
    """

    def wrapper(*args, **kwargs):
        header_dict = header_func(*args, **kwargs)

        # Only check dimension one time
        if not wrapper.dimension_checked:
            # Check only relevant to the shell model
            if cfg.MODEL == Models.SHELL_MODEL:
                # Check if dimension in header dict matches dimension in params
                try:
                    if header_dict["sdim"] != PAR_SH.sdim:
                        sh_utils.set_params(PAR_SH, sdim=header_dict["sdim"])
                        print(
                            f"{col.Fore.GREEN}INFO: sdim param updated to match header_dict{col.Fore.RESET}\n"
                        )
                        # Update arrays
                        sh_utils.update_arrays(PAR_SH)

                except KeyError:
                    print(
                        f"{col.Fore.RED}sdim not found in header_dict. Using sdim={PAR_SH.sdim} from parameter struct{col.Fore.RESET}"
                    )

                wrapper.dimension_checked = True

        return header_dict

    wrapper.dimension_checked = False

    return wrapper
