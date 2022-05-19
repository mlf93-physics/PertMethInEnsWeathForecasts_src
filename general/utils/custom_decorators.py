import numpy as np
import colorama as col
from general.params.model_licences import Models
import config as cfg

if cfg.MODEL == Models.SHELL_MODEL:
    # Shell model specific imports
    import shell_model_experiments.utils.util_funcs as sh_utils
    import shell_model_experiments.utils.special_params as sh_sparams
    from shell_model_experiments.params.params import PAR, ParamsStructType

    # Get parameters for model
    params = PAR
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    # Get parameters for model
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams

    # Get parameters for model
    params = l63_params
    sparams = l63_sparams


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
                    if header_dict["sdim"] != params.sdim:
                        sh_utils.set_params(params, sdim=header_dict["sdim"])
                        print(
                            f"{col.Fore.GREEN}INFO: sdim param updated to match header_dict{col.Fore.RESET}\n"
                        )
                        # Update arrays
                        sh_utils.update_arrays(params)

                except KeyError:
                    print(
                        f"{col.Fore.RED}sdim not found in header_dict. Using sdim={params.sdim} from parameter struct{col.Fore.RESET}"
                    )

                wrapper.dimension_checked = True

        return header_dict

    wrapper.dimension_checked = False

    return wrapper


def cached_profiles(header_func: callable) -> callable:
    def wrapper(*args, **kwargs):
        if wrapper.counter > 0:
            if "start_times" in kwargs.keys():
                len_start_times = len(kwargs["start_times"])
                if len_start_times > 0 and len_start_times == len(wrapper.positions):
                    # Check if saved positions correspond to requested start times
                    if (
                        all(
                            [
                                int(round(kwargs["start_times"][i] * params.tts))
                                == wrapper.positions[i]
                                for i in range(len(kwargs["start_times"]))
                            ]
                        )
                        and kwargs["args"]["n_runs_per_profile"]
                        == wrapper.n_runs_per_profile
                    ):
                        return (
                            wrapper.u_init_profiles,
                            wrapper.positions,
                            wrapper.ref_header_dict,
                        )
            elif kwargs["args"]["start_times"] is not None:
                len_args_start_times = len(kwargs["args"]["start_times"])
                if len_args_start_times > 0 and len_args_start_times == len(
                    wrapper.positions
                ):
                    # Check if saved positions correspond to requested start times
                    if (
                        all(
                            [
                                int(
                                    round(kwargs["args"]["start_times"][i] * params.tts)
                                )
                                == wrapper.positions[i]
                                for i in range(len_args_start_times)
                            ]
                        )
                        and kwargs["args"]["n_runs_per_profile"]
                        == wrapper.n_runs_per_profile
                    ):
                        return (
                            wrapper.u_init_profiles,
                            wrapper.positions,
                            wrapper.ref_header_dict,
                        )

        # Import profiles if not using cached profiles
        (u_init_profiles, positions, ref_header_dict) = header_func(*args, **kwargs)

        wrapper.u_init_profiles = u_init_profiles
        wrapper.positions = positions
        wrapper.ref_header_dict = ref_header_dict
        wrapper.n_runs_per_profile = kwargs["args"]["n_runs_per_profile"]

        wrapper.counter += 1

        return (
            wrapper.u_init_profiles,
            wrapper.positions,
            wrapper.ref_header_dict,
        )

    wrapper.counter = 0
    wrapper.u_init_profiles = None
    wrapper.positions = None
    wrapper.ref_header_dict = None
    wrapper.n_runs_per_profile = None

    return wrapper
