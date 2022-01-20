import os
import sys

sys.path.append("..")
import numpy as np
import pathlib as pl
import decimal
import general.utils.importing.import_data_funcs as g_import
import general.utils.util_funcs as g_utils
import general.utils.importing.import_perturbation_data as pt_import
import general.utils.perturb_utils as pt_utils
from general.params.experiment_licences import Experiments as EXP
from general.utils.module_import.type_import import *
import general.utils.exceptions as g_exceptions
import config as cfg
from general.params.model_licences import Models

if cfg.MODEL == Models.SHELL_MODEL:
    # Shell model specific imports
    import shell_model_experiments.utils.special_params as sh_sparams
    from shell_model_experiments.params.params import PAR
    from shell_model_experiments.params.params import ParamsStructType
    import shell_model_experiments.perturbations.normal_modes as sh_nm_estimator

    # Get parameters for model
    params = PAR
    sparams = sh_sparams
elif cfg.MODEL == Models.LORENTZ63:
    # Lorentz-63 model specific imports
    import lorentz63_experiments.params.params as l63_params
    import lorentz63_experiments.params.special_params as l63_sparams
    import lorentz63_experiments.perturbations.normal_modes as l63_nm_estimator

    # Get parameters for model
    params = l63_params
    sparams = l63_sparams


def get_bv_start_time(
    eval_time: Union[float, np.ndarray] = None, exp_setup: dict = None
):
    _temp_offset = exp_setup["n_cycles"] * exp_setup["integration_time"]
    if _temp_offset > np.min(eval_time):
        raise ValueError(
            "The BV offset (calculated as n_cycles * integration_time) is to "
            + "large compared to the eval_time"
        )

    start_time = eval_time - _temp_offset
    return start_time


def generate_start_times(exp_setup: dict, args: dict):
    """Generate start times and calculate the number of possible units from
    the relevant run-time variables and variables from the experiment setup

    Parameters
    ----------
    exp_setup : dict
        The current experiment setup
    args : dict
        Run-time arguments

    Returns
    -------
    tuple
        All generated start times and the number of possible units/start times
        (
            list: start_times
            int: num_possible_units
        )

    Raises
    ------
    g_exceptions.LicenceImplementationError
        Raised if the present function do not work on the current licence
    """
    ref_header_dict = g_import.import_info_file(pl.Path(args["datapath"], "ref_data"))

    if cfg.LICENCE == EXP.LORENTZ_BLOCK:
        offset_var = "block_offset"
    elif (
        cfg.LICENCE == EXP.BREEDING_VECTORS
        or cfg.LICENCE == EXP.LYAPUNOV_VECTORS
        or cfg.LICENCE == EXP.SINGULAR_VECTORS
    ):
        offset_var = "vector_offset"
    else:
        raise g_exceptions.LicenceImplementationError(licence=cfg.LICENCE)

    if offset_var in exp_setup:
        if "start_times" in exp_setup:
            _time_offset = exp_setup["start_times"][0]
        elif "eval_times" in exp_setup:
            if cfg.LICENCE == EXP.BREEDING_VECTORS:
                _time_offset = get_bv_start_time(
                    eval_time=exp_setup["eval_times"][0], exp_setup=exp_setup
                )
            elif cfg.LICENCE == EXP.LYAPUNOV_VECTORS:
                _time_offset = (
                    exp_setup["eval_times"][0] - exp_setup["integration_time"]
                )
            elif cfg.LICENCE == EXP.SINGULAR_VECTORS:
                _time_offset = exp_setup["eval_times"][0]
            else:
                raise g_exceptions.LicenceImplementationError(licence=cfg.LICENCE)
        else:
            _time_offset = 0

        if "integration_time" in exp_setup:
            _time_to_run = exp_setup["integration_time"]
        elif "time_to_run" in exp_setup:
            _time_to_run = exp_setup["time_to_run"]
        else:
            raise ValueError("Could not infer time_to_run from experiment setup")

        # Determine precision of time
        _precision = decimal.Decimal(str(_time_to_run)).as_tuple().exponent

        num_possible_units = int(
            (ref_header_dict["time_to_run"] - _time_offset) // exp_setup[offset_var]
        )
        # Calculate start_times and round off correctly
        start_times = [
            round(exp_setup[offset_var] * i + _time_offset, abs(_precision))
            for i in range(num_possible_units)
        ]
    elif "start_times" in exp_setup:
        num_possible_units = len(exp_setup["start_times"])
        start_times = exp_setup["start_times"]

    return start_times, num_possible_units


def adjust_run_setup(args: dict):
    # Disable stdout
    if args["noprint"]:
        sys.stdout = open(os.devnull, "w")


def prepare_run_times(
    args: dict,
) -> Tuple[np.ndarray, np.ndarray]:

    num_perturbations = args["n_runs_per_profile"] * args["n_profiles"]

    if cfg.LICENCE != EXP.LORENTZ_BLOCK:
        times_to_run = np.ones(num_perturbations) * args["time_to_run"]
        Nt_array = np.round(times_to_run / params.dt).astype(np.int64)
    else:
        if len(args["start_times"]) > 1:
            start_times = np.array(args["start_times"])
        else:
            start_times = np.ones(num_perturbations) * args["start_times"]

        times_to_run = start_times[0] + args["time_to_run"] - start_times
        Nt_array = (times_to_run / params.dt).astype(np.int64)

    return times_to_run, Nt_array


def prepare_perturbations(
    args: dict, raw_perturbations: bool = False
) -> Tuple[np.ndarray, List[int]]:
    """Prepare the perturbed initial conditions according to the desired perturbation
    type (given by args["pert_mode"])

    Parameters
    ----------
    args : dict
        Run-time arguments
    raw_perturbations : bool, optional
        If the raw perturbations should be returned instead of the perturbations
        added to the u_init_profiles, by default False

    Returns
    -------
    tuple
        (
            u_return : perturbed u profiles
            perturb_positions : position of the perturbation (in samples)
        )


    Raises
    ------
    g_exceptions.ModelError
        Raised if the single_shell_perturb option is set while not using the shell
        model
    g_exceptions.InvalidRuntimeArgument
        Raised if the perturbation mode is not valid
    """

    # Import reference info file
    ref_header_dict = g_import.import_info_file(pl.Path(args["datapath"], "ref_data"))
    # header_dict = g_utils.handle_different_headers(header_dict)

    # Adjust parameters to have the correct ny/ny_n for shell model
    g_utils.determine_params_from_header_dict(ref_header_dict, args)

    # Only import start profiles beforehand if not using bv, bv_eof or sv perturbations,
    # i.e. also when running in singel_shell_perturb mode

    if args["pert_mode"] not in ["bv", "bv_eof", "sv"]:
        (
            u_init_profiles,
            perturb_positions,
            header_dict,
        ) = g_import.import_start_u_profiles(args=args)

    # NM pert generation mode; if item in array is True, all perturbations are
    # run for a given profile - perturbations lie in the plane of the
    # complex-conjugate pair of the leading NM. Otherwise only one run per
    # profile is executed. Defaults to None for all other types of perturbations
    exec_all_runs_per_profile: Union[np.ndarray, None] = None

    if args["pert_mode"] is not None:

        if args["pert_mode"] == "nm":
            print("\nRunning with NORMAL MODE perturbations\n")
            if cfg.MODEL == Models.SHELL_MODEL:
                (perturb_vectors, _, _,) = sh_nm_estimator.find_normal_modes(
                    u_init_profiles[
                        :,
                        0 : args["n_profiles"]
                        * args["n_runs_per_profile"] : args["n_runs_per_profile"],
                    ],
                    args,
                    dev_plot_active=False,
                    local_ny=header_dict["ny"],
                )
            elif cfg.MODEL == Models.LORENTZ63:
                (
                    perturb_vectors,
                    e_values_max,
                    _,
                    _,
                ) = l63_nm_estimator.find_normal_modes(
                    u_init_profiles[
                        :,
                        0 : args["n_profiles"]
                        * args["n_runs_per_profile"] : args["n_runs_per_profile"],
                    ],
                    args,
                    n_profiles=args["n_profiles"],
                )
                # Evaluate if e_values have imaginary part. Used to determine if
                # all runs_per_profile or just one run should be executed (see
                # L. Magnusson 2008)
                exec_all_runs_per_profile = e_values_max.imag != 0
        elif "bv" in args["pert_mode"]:
            if args["pert_mode"] == "bv":
                print("\nRunning with BREED VECTOR perturbations\n")
                _raw_perturbations = False
            elif args["pert_mode"] == "bv_eof":
                print("\nRunning with BREED EOF VECTOR perturbations\n")
                _raw_perturbations = True
            (
                perturb_vectors,
                _,
                u_init_profiles,
                perturb_positions,
                _,
            ) = pt_import.import_perturb_vectors(
                args, raw_perturbations=_raw_perturbations
            )

            # Reshape perturb_vectors
            perturb_vectors = np.reshape(
                np.transpose(perturb_vectors, axes=(2, 0, 1)),
                (params.sdim, args["n_profiles"] * args["n_runs_per_profile"]),
            )

        elif "sv" in args["pert_mode"]:
            print("\nRunning with SINGULAR VECTOR perturbations\n")
            (
                perturb_vectors,
                _,
                u_init_profiles,
                perturb_positions,
                _,
            ) = pt_import.import_perturb_vectors(
                args, raw_perturbations=True, dtype=np.complex128
            )
            # Reshape perturb_vectors
            perturb_vectors = np.reshape(
                np.transpose(perturb_vectors, axes=(2, 0, 1)),
                (params.sdim, args["n_profiles"] * args["n_runs_per_profile"]),
            )

        elif args["pert_mode"] == "rd":
            print("\nRunning with RANDOM perturbations\n")
            perturb_vectors = np.ones(
                (params.sdim, args["n_profiles"]), dtype=sparams.dtype
            )

        elif args["pert_mode"] == "rf":
            perturb_vectors = pt_utils.get_rand_field_perturbations(
                args,
                u_init_profiles=u_init_profiles,
                start_times=perturb_positions * params.stt,
            )

    # Check if single shell perturb should be activated
    elif args["single_shell_perturb"] is not None:
        # Specific to shell model setup
        if cfg.MODEL == Models.SHELL_MODEL:
            print("\nRunning in single shell perturb mode\n")
            perturb_vectors = None
        else:
            raise g_exceptions.ModelError(model=cfg.MODEL)
    else:
        _pert_arg = (
            "pert_mode: " + args["pert_mode"]
            if "pert_mode" in args
            else "single_shell_perturb: " + args["single_shell_perturb"]
        )
        raise g_exceptions.InvalidRuntimeArgument(
            "Not a valid perturbation mode", argument=_pert_arg
        )

    # Make perturbations
    perturbations = pt_utils.calculate_perturbations(
        perturb_vectors, dev_plot_active=False, args=args
    )

    if raw_perturbations:
        # Return raw perturbations
        u_return = perturbations
    else:
        # Apply perturbations
        u_return = u_init_profiles + perturbations

    return u_return, perturb_positions, exec_all_runs_per_profile
