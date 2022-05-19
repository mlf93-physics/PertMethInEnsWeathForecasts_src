__all__ = [
    "LINESTYLES",
    "FIG_ROOT",
    "MARKERS",
    "SUBFIG_LABELS",
    "METHOD_COLORS",
    "METHOD_LINESTYLES",
    "LINEWIDTHS",
]

import pathlib as pl
import seaborn as sb

MARKERS = [".", "v", "s", "+", "x", "^", "*", "p", "d"]

FIG_ROOT = pl.Path("/home/martin/Dropbox/FysikUni/MasterThesis/thesis/figures")

SUBFIG_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

METHOD_COLORS = {
    "rd": sb.color_palette()[5],  # brown
    "nm": sb.color_palette()[4],  # purple
    "bv": sb.color_palette()[0],  # blue
    "bv_eof": sb.color_palette()[1],  # orange
    "sv": sb.color_palette()[2],  # green
    "lv": sb.color_palette()[3],  # red
    "rf": sb.color_palette()[6],  # pink
}

METHOD_LINESTYLES = {
    "bv_eof": ["-", "--", "-.", ":"],
    "sv": ["-", "--", "-.", ":"],
    "lv": ["-", "--", "-.", ":"],
}

LINEWIDTHS = {"thin": 1.0, "medium": 1.5}

LINESTYLES = [
    "-",
    "--",
    "-.",
    ":",
    "-",
    "--",
    "-.",
    ":",
    "-",
    "--",
    "-.",
    ":",
    "-",
    "--",
    "-.",
    ":",
    "-",
    "--",
    "-.",
    ":",
    "-",
    "--",
    "-.",
    ":",
    "-",
    "--",
    "-.",
    ":",
    "-",
    "--",
    "-.",
    ":",
    "-",
    "--",
    "-.",
    ":",
]
