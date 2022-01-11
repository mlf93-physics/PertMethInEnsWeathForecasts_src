import sys

sys.path.append("..")
import numpy as np
from general.utils.module_import.type_import import *
import general.utils.importing.import_data_funcs as g_import
import general.utils.argument_parsers as a_parsers
import general.utils.user_interface as g_ui
import general.utils.util_funcs as g_utils
from lorentz63_experiments.params.params import *
import config as cfg


def choose_rand_field_indices(
    u_data: np.ndarray,
) -> Generator[Tuple[int, int], None, None]:
    # Define constants
    dt_wing = 1.8  # mean time spent in one wing (L. Magnusson et al, 2008
    corr_time = 4 * dt_wing
    corr_dindex = int(corr_time * tts)
    n_datapoints = u_data.shape[0]

    # Define arrays
    wing_bool_array = u_data[:, 0] > 0
    data_indices = np.arange(0, n_datapoints)

    def iterator() -> Tuple[int, int]:
        """Get index pair for two random fields belonging to the same attractor
        wing and minimum separated by the time corr_time

        Returns
        -------
        Tuple(int, int)
            (
                rand_field1_index : The index of the random field1
                rand_field2_index : The index of the random field2
            )
        """
        # Get random index of field1
        rand_field1_index = np.random.randint(low=0, high=n_datapoints)
        # Determine what wing the index belongs to
        rand_field1_wing = wing_bool_array[rand_field1_index]

        # Collect all possible indices for random field2
        rand_field2_indices = np.concatenate(
            [
                data_indices[: max((rand_field1_index - corr_dindex), 0)],
                data_indices[min(rand_field1_index + corr_dindex, n_datapoints - 1) :],
            ]
        )

        # Make shure that the random field2 index belongs to the same wing as field1
        rand_field2_indices = np.intersect1d(
            rand_field2_indices, np.argwhere(wing_bool_array == rand_field1_wing)
        )
        # Choose the random field2 index
        rand_field2_index = np.random.choice(rand_field2_indices, size=1)

        return rand_field1_index, rand_field2_index

    while True:
        rand_field1_index, rand_field2_index = iterator()
        yield rand_field1_index, rand_field2_index


# if __name__ == "__main__":
#     cfg.init_licence()

#     # Get arguments
#     _parser = a_parsers.PerturbationArgSetup()
#     _parser.setup_parser()
#     _parser.validate_arguments()
#     _parser = a_parsers.ReferenceAnalysisArgParser()
#     _parser.setup_parser()
#     args = _parser.args

#     g_ui.confirm_run_setup(args)

#     get_rand_field_perturbations(args)
