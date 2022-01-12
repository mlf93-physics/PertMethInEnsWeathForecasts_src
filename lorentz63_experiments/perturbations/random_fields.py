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
    u_data: np.ndarray, wing_u_init_profiles: np.ndarray
) -> Generator[Tuple[int, int], None, None]:
    # Define constants
    dt_wing = 1.8  # mean time spent in one wing (L. Magnusson et al, 2008
    corr_time = 0  # 20 * dt_wing
    corr_dindex = int(corr_time * tts)
    n_datapoints = u_data.shape[0]

    # Define arrays
    wing_bool_array = u_data[:, 0] > 0
    # data_indices = np.arange(0, n_datapoints)

    def iterator(wing_u_init: bool) -> Tuple[int, int]:
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
        # Get the indices of the wing to which the random fields shall belong
        wing_indices = np.argwhere(wing_bool_array == wing_u_init).ravel()
        # Get random index of field1
        rand_field1_index = np.random.choice(wing_indices)

        # Collect all possible indices for random field2
        # rand_field2_indices = np.concatenate(
        #     [
        #         data_indices[: max((rand_field1_index - corr_dindex), 0)],
        #         data_indices[min(rand_field1_index + corr_dindex, n_datapoints - 1) :],
        #     ]
        # )

        # # Make shure that the random field2 index belongs to the same wing as field1
        # rand_field2_indices = np.intersect1d(rand_field2_indices, wing_indices)
        # Choose the random field2 index
        rand_field2_index = np.random.choice(wing_indices)

        return rand_field1_index, rand_field2_index

    counter = 0
    while True:
        rand_field1_index, rand_field2_index = iterator(wing_u_init_profiles[counter])
        counter += 1

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
