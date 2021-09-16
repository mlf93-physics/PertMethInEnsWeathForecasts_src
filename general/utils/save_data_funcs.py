import os
import pathlib as pl
import subprocess as sp


def generate_dir(expected_path, subfolder="", args=None):

    if len(subfolder) == 0:
        # See if folder is present
        dir_exists = os.path.isdir(expected_path)

        if not dir_exists:
            os.makedirs(expected_path)

        subfolder = expected_path
    else:
        # Check if path exists
        expected_path = str(pl.Path(expected_path, subfolder))
        dir_exists = os.path.isdir(expected_path)

        if not dir_exists:
            os.makedirs(expected_path)

    return expected_path


def compress_dir(path_to_dir, zip_name):
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


if __name__ == "__main__":
    dir = "./data/ny2.37e-08_t4.00e+02_n_f0_f1.0/lorentz_block_short_pred_ttr0.25/"
    zip_name = "test_tar"

    compress_dir(dir, zip_name)
