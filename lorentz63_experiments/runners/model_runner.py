import os
import sys

sys.path.append("..")
import argparse
from pyinstrument import Profiler

profiler = Profiler()


def model_runner(args):
    print("Running Lorentz63 model")


if __name__ == "__main__":
    # Define arguments
    arg_parser = argparse.ArgumentParser()
    args = arg_parser.parse_args()

    model_runner(args)
