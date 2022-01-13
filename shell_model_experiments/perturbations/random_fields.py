import sys

sys.path.append("..")
import numpy as np
from general.utils.module_import.type_import import *
from shell_model_experiments.params.params import PAR
from shell_model_experiments.params.params import ParamsStructType
import config as cfg
import general.utils.argument_parsers as a_parsers
import general.utils.importing.import_data_funcs as g_import
import shell_model_experiments.utils.runner_utils as sh_r_utils
import shell_model_experiments.utils.util_funcs as ut_funcs


def choose_rand_field_indices(
    regime_start_time_indices: np.ndarray, args: dict, regime_start_time_header: dict
) -> Generator[Tuple[int, int], None, None]:
    # Define constants
    num_fields = 2
    corr_time = 3  # one eddy turnover time of first shell
    corr_dindex = int(corr_time * PAR.tts)
    num_start_times = regime_start_time_indices.shape[0]
    rand_field_indices = np.empty(2, dtype=np.int64)

    def iterator() -> Tuple[int, int]:
        """Get index pair for two random fields belonging to the same attractor
        wing as the u_init_profile (determined by wing_u_init) and minimum
        separated by the time corr_time

        Parameters
        ----------
        wing_u_init : bool
            The wing to which the random fields shall belong (corresponding to
            the wing of the u_init_profile)

        Returns
        -------
        Tuple(int, int)
            (
                rand_field1_index : The index of the random field1
                rand_field2_index : The index of the random field2
            )
        """

        for i in range(num_fields):
            # Choose randomly which region (between two start times) to sample from
            # NOTE: "high" argument is exclusive, which is needed because last index
            # is not possible to use if regime_start = low, since no region endpoint
            # (in the form of a start time for the following high pred regime)
            # exists
            regime_region_index = np.random.randint(low=0, high=num_start_times)

            if args["regime_start"] is None:
                raise ValueError(
                    "regime_start is None, but it is needed to select random fields"
                )
            else:
                # Prepare variable for indexing regime_start_time_indices array
                _start_time_array_index = int(
                    # Use header value for index
                    regime_start_time_header[args["regime_start"]]
                )
                # Choose random index in the region
                rand_field_indices[i] = np.random.randint(
                    low=regime_start_time_indices[
                        regime_region_index, _start_time_array_index
                    ],
                    high=regime_start_time_indices[
                        regime_region_index + (_start_time_array_index + 1) // 2,
                        (_start_time_array_index + 1) % 2,
                    ],
                )

        return rand_field_indices[0], rand_field_indices[1]

    counter = 0
    while True:
        rand_field1_index, rand_field2_index = iterator()
        counter += 1

        yield rand_field1_index, rand_field2_index


if __name__ == "__main__":
    cfg.init_licence()

    # Get arguments
    mult_pert_arg_setup = a_parsers.MultiPerturbationArgSetup()
    mult_pert_arg_setup.setup_parser()
    ref_arg_setup = a_parsers.ReferenceAnalysisArgParser()
    ref_arg_setup.setup_parser()
    args = ref_arg_setup.args

    # Initiate and update variables and arrays
    ut_funcs.update_dependent_params(PAR)
    ut_funcs.update_arrays(PAR)

    # time, u_data, ref_header_dict = g_import.import_ref_data(args=args)

    start_times, num_start_times, header = sh_r_utils.get_regime_start_times(
        args, return_all=True
    )

    # Convert start_times to indices
    regime_start_time_indices = (start_times * PAR.tts).astype(np.int64)

    # Define iterator
    iterator = choose_rand_field_indices(regime_start_time_indices, args, header)
    for _ in range(args["n_profiles"]):
        rand_field1_index, rand_field2_index = next(iterator)

        print(
            "rand_field1_index, rand_field2_index", rand_field1_index, rand_field2_index
        )
