import sys

sys.path.append("..")
import shell_model_experiments.utils.util_funcs as sh_utils


def diffusion_type_decorator(func):
    def wrapper(*args, diff_type="normal", **kwargs):
        if diff_type == "normal":
            u_old = func(sh_utils.normal_diffusion, *args, **kwargs)
        elif diff_type == "inf_hyper":
            u_old = func(sh_utils.infinit_hyper_diffusion, *args, **kwargs)

        return u_old

    return wrapper
