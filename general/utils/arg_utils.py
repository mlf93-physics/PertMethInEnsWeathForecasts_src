import copy
from general.utils.argument_parsers import PT_CHOICES, V_CHOICES


def react_on_comparison_arguments(args):
    if len(args["perturbations"]) == 1 and args["perturbations"][0] == "all":

        args["perturbations"] = copy.deepcopy(PT_CHOICES)
        args["perturbations"].remove("all")
        args["perturbations"].remove("tl_rd")
    if len(args["vectors"]) == 1 and args["vectors"][0] == "all":
        args["vectors"] = copy.deepcopy(V_CHOICES)
        args["vectors"].remove("all")
