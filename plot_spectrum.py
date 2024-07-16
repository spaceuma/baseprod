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
Plot individual spectra.

Call the function with:
python plot_spectrum --file <csv> --spectrum <column in the csv>

Get an overview of the options with:
    python plot_spectrum --help

The spectrum file is 'spectra.csv'.
"""

__author__ = "Joaquín Ortega Cortés"


import argparse
import csv

import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = []
    y = []

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file", type=str, required=True, help="the file of spectra to plot"
    )
    parser.add_argument(
        "--spectrum",
        type=int,
        required=True,
        help="the number of the spectrum to plot inside the spectra",
    )
    parser.add_argument(
        "--latex",
        "-l",
        action="store_true",
        dest="latex",
        help="Render text in Latex",
    )
    args = parser.parse_args()

    plt.rcParams["svg.fonttype"] = "none"
    if args.latex:
        plt.rcParams.update(
            {"text.usetex": True, "font.family": "Computer Modern Roman"}
        )

    with open(args.file, "r") as csvfile:
        plots = csv.reader(csvfile, delimiter="\t")
        for row in plots:
            x.append(float(row[0]))
            y.append(float(row[args.spectrum]))

        plt.plot(x, y, color="g", label="Spectrum")

        plt.xticks(rotation=25)
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Intensity [counts]")
        plt.title("Spectrum", fontsize=20)
        plt.grid()
        plt.legend()
        plt.show()
