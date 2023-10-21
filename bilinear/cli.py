"""
Bilinear time-then-space interpolation for jittery, sparse camera arrays.

* Naming: {angle:int}-{time:float}.{ext}
* Order: Ascending right-to-left or --reverse
"""
import argparse


show_defaults = argparse.ArgumentDefaultsHelpFormatter
pre_wrap = argparse.RawTextHelpFormatter


class Format(show_defaults, pre_wrap):
    pass


parser = argparse.ArgumentParser(description=__doc__, formatter_class=Format)

parser.add_argument("--name", type=str, default="quilt", help="Quilt video prefix")
parser.add_argument("--input", type=str, default="mesh", help="Input image folder")
parser.add_argument("--ext", type=str, default="png", help="Input image extension")
parser.add_argument("--fps", type=int, default=24, help="Temporal resolution")
parser.add_argument("--dim", type=int, default=48, help="Spatial resolution")
parser.add_argument("--qw", type=int, default=8, help="Quilt width")
parser.add_argument("--reverse", action="store_true", help="Left to right angles")

args = parser.parse_args()
