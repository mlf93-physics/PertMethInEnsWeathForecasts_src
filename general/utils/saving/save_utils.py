import os
import pathlib as pl
import subprocess as sp
from shell_model_experiments.params.params import PAR as PAR_SH
from shell_model_experiments.params.params import ParamsStructType
import lorentz63_experiments.params.params as l63_params
from general.params.model_licences import Models
import config as cfg


def convert_arguments_to_string(args):
    """Convert specific arguments to string format using .format

    Parameters
    ----------
    args : dict
        A dictionary containing run-time arguments

    """

    arguments = {}

    # Some universal argument conversion
    arguments["time_to_run"] = "{:.2e}".format(args["time_to_run"])

    if cfg.MODEL == Models.SHELL_MODEL:
        arguments["forcing"] = "{:.1f}".format(args["forcing"])
        arguments["ny"] = "{:.2e}".format(args["ny"])
    elif cfg.MODEL == Models.LORENTZ63:
        arguments["sigma"] = "{:.2e}".format(args["sigma"])
        arguments["r_const"] = "{:.2e}".format(args["r_const"])
        arguments["b_const"] = "{:.2e}".format(args["b_const"])

    return arguments


def generate_standard_data_name(args):

    adj_args = convert_arguments_to_string(args)

    if cfg.MODEL == Models.SHELL_MODEL:
        file_name = (
            f"ny{adj_args['ny']}_ny_n{args['ny_n']}_t{adj_args['time_to_run']}"
            + f"_n_f{PAR_SH.n_forcing}_f{adj_args['forcing']}"
            f"_kexp{args['diff_exponent']}"
        )
    elif cfg.MODEL == Models.LORENTZ63:
        file_name = (
            f"sig{adj_args['sigma']}_t{adj_args['time_to_run']}"
            + f"_b{adj_args['b_const']}_r{adj_args['r_const']}_dt{l63_params.dt}"
        )

    return file_name


def args_to_string(args):
    """Convert args dictionary to string

    Parameters
    ----------
    args : dict
        A dictionary containing run-time arguments

    """
    if args is None:
        return ""

    arg_str_list = [f"{key}={value}" for key, value in args.items()]
    arg_str = ", ".join(arg_str_list)

    return arg_str


def generate_dir(expected_path, subfolder="", args=None):
    """Generate a directory from a path and possibly a subfolder

    Parameters
    ----------
    expected_path : str, Path
        Path to the dir
    subfolder : str, optional
        Subfolder to append to the path
    args : dict
        A dictionary containing run-time arguments

    """

    if len(subfolder) == 0:
        # See if folder is present
        dir_exists = os.path.isdir(expected_path)

        if not dir_exists:
            os.makedirs(expected_path)

    else:
        # Check if path exists
        expected_path = str(pl.Path(expected_path, subfolder))
        dir_exists = os.path.isdir(expected_path)

        if not dir_exists:
            os.makedirs(expected_path)

    return expected_path


def compress_dir(path_to_dir: pl.Path, zip_name: str = "test_temp1"):
    """Compress a directory using tar. The resulting .tar.gz file will be
    located at path_to_dir

    Parameters
    ----------
    path_to_dir : str, Path
        Path to the directory/file to compress
    zip_name : str
        The name of the produced zip

    """
    if not os.path.isdir(path_to_dir):
        raise ValueError(f"No dir at the given path ({path_to_dir})")
    else:
        path_to_dir = pl.Path(path_to_dir)

    out_name = pl.Path(path_to_dir.parent, zip_name + ".tar.gz")

    print(f"Compressing data directory at path: {path_to_dir}")
    sp.run(
        ["tar", "-czvf", str(out_name), "-C", path_to_dir.parent, path_to_dir.name],
        stdout=sp.DEVNULL,
    )


def generate_header(
    args: dict, n_data: int = 0, append_extra: str = "", append_options: list = []
):
    """Generate a header string

    Parameters
    ----------
    args : dict
        A dictionary containing run-time arguments
    n_data : int
        Number of datapoints
    append_extra : str
        Some string to append to the header

    """

    header = args_to_string(args)

    if cfg.MODEL == Models.SHELL_MODEL:
        header += (
            f", n_f={PAR_SH.n_forcing}, dt={PAR_SH.dt}, epsilon={PAR_SH.epsilon}, "
            + f"lambda={PAR_SH.lambda_const}, N_data={n_data}, "
            + f"sample_rate={PAR_SH.sample_rate}, "
        )
    elif cfg.MODEL == Models.LORENTZ63:
        header += (
            f", dt={l63_params.dt}, N_data={n_data}, "
            + f"sample_rate={l63_params.sample_rate}, "
        )

    optional_append = ""
    if "licence" in append_options:
        optional_append += f"experiment={cfg.LICENCE}, "

    header += optional_append + append_extra

    # Strip trailing commas
    header = header.rstrip(",")
    header = header.rstrip(", ")
    # Remove \n symbols
    header = header.replace("\n", "")

    return header
