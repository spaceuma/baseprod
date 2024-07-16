# MIT License
#
# Copyright (c) 2024 Space Robotics Lab at UMA
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Correct F/T values in input path.

The script uses pre-computed slopes, computes offsets from an input calibration file,
and corrects values it reads from an input path.
The corrected values are saved in new files next to the input files.
"""

__author__ = "Levin Gerdes"


import argparse
import os

import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bogie_offset import to_datenum


def get_args() -> argparse.Namespace:
    """Parses CLI arguments"""

    parser = argparse.ArgumentParser(description="")

    # fmt: off
    parser.add_argument("--input-path", "-i", type=str, default="/media/srl/Nuevo vol/bardenas/2023-07-21_17-45-42/",   dest="input_path",       help="Path of directory containing the CSV to be corrected (will only be read), e.g. '/path/to/dataset/2023-07-21_17-45-42'")
    parser.add_argument("--cal-path",   "-c", type=str, default="/media/srl/Nuevo vol/calibration/ft_2023-07-20_11-52/", dest="calibration_path", help="Offset calibration path (folder with FTS CSVs)")
    parser.add_argument("--precision",  "-p", type=int, default=12, dest="precision", help="Num. of digits behind decimal point in CSV")
    parser.add_argument("--quiet", "-q", action="store_false", dest="show_fig", help="Don't show figure. E.g. if you want to run this in batch")
    parser.add_argument("--latex", "-l", action="store_true", dest="latex", help="Render text in Latex")
    # fmt: on

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    FTS = ["FL", "FR", "CL", "CR", "BL", "BR"]
    FIELDS = ["Force_X", "Force_Y", "Force_Z", "Torque_X", "Torque_Y", "Torque_Z"]

    plt.rcParams["svg.fonttype"] = "none"
    if args.latex:
        # requires the following packages:
        # sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
        plt.rcParams.update(
            {"text.usetex": True, "font.family": "Computer Modern Roman"}
        )
    xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")

    # The slope correction uses 'ft_slopes.py' and assumes that the slope correction
    # needed for the force (computed) is the same for the torques. We suspect the
    # correction to be necessary because of the cable length, which stays the same
    # for the individual sensor of course, regardless of whether we read forces or torques.
    ideal_slope = 9.81
    slope_correction = {
        "FL": ideal_slope / 9.692683660932358,
        "FR": ideal_slope / 9.46239216545322,
        "CL": ideal_slope / 9.493103012577023,
        "CR": ideal_slope / 9.368096142864271,
        "BL": ideal_slope / 9.531557477873573,
        "BR": ideal_slope / 9.651076358070236,
    }

    for i, fts in enumerate(FTS):
        # Calibration file is used to find the offsets only
        df_cal = pd.read_csv(os.path.join(args.calibration_path, f"FTS_{fts}.csv"))
        # Recording to be corrected (in a copy)
        df_rec = pd.read_csv(os.path.join(args.input_path, f"FTS_{fts}.csv"))
        for f in FIELDS:
            # Select subplot for current values
            # subplot indices start at 1
            plt.figure(i + 1)
            subplot_number = 1 if "Force" in f else 2
            plt.subplot(2, 1, subplot_number)

            print(f"Mean for {fts} {f} recorded: {np.mean(df_rec[f])}")
            plt.plot(
                to_datenum((df_rec["Timestamp"] / 1e9).tolist()),
                df_rec[f],
                "--",
                label=f"{fts} {f} recorded",
                linewidth=1,
            )

            df_rec[f] = (df_rec[f] - np.mean(df_cal[f])) * slope_correction[fts]
            df_rec[f] = [
                float(np.format_float_positional(x, args.precision, trim="-"))
                for x in df_rec[f]
            ]

            print(f"Mean for {fts} {f} corrected: {np.mean(df_rec[f])}")
            plt.plot(
                to_datenum((df_rec["Timestamp"] / 1e9).tolist()),
                df_rec[f],
                "-",
                label=f"{fts} {f} corrected",
                linewidth=1,
            )
        df_rec.to_csv(
            os.path.join(args.input_path, f"FTS_{fts}_CORRECTED.csv"), index=False
        )

    # Plot labels
    for i, fts in enumerate(FTS):
        for f in FIELDS:
            plt.figure(i + 1)
            plt.xlabel("Time")
            plt.subplots_adjust(bottom=0.2)
            plt.xticks(rotation=25)
            ax = plt.gca()
            ax.xaxis.set_major_formatter(xfmt)
            if "Force" in f:
                plt.subplot(2, 1, 1)
                plt.ylabel("Force [N]")
            else:
                plt.subplot(2, 1, 2)
                plt.ylabel("Torque [Nm]")
            plt.legend()
        plt.savefig(
            os.path.join(args.input_path, f"FTS_{fts}_corrections.svg"),
            format="svg",
        )

    if args.show_fig:
        plt.show()
