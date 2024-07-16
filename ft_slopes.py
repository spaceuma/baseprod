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
Force-Torque sensor slope correction.

An output svg figure can be saved by providing an output flag, e.g.,
python ft_slopes.py -o ft_slopes.svg

This script contains the measurements of all FT sensors.
During the first measurements, no force was applied,
then weights of up to about 10kg were added and the output measured.
In the figure there is a thick red line which represents the ideal behaviour of 9.81N/kg.
"""

__author__ = "Levin Gerdes and Felix Wilting"


import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression  # type: ignore

X, Y, Z = {}, {}, {}
# fmt:off
W       =      [0, 0.4772, 0.9276, 1.4048, 1.9236, 2.4008, 3.1472, 3.6244, 4.0748,  4.552, 5.0708,  5.548, 6.2672, 6.7444, 7.1948,  7.672, 8.1908,  8.668, 9.2064, 9.6836, 10.134]
X["BL"] =  [-20.2,  -20.1,  -20.1,  -20.1,  -20.1,  -20.1,  -20.2,  -20.2,  -20.2,  -20.2,  -20.3,   20.3,  -20.5,  -20.5,  -20.5,  -20.5,  -20.4,  -20.3,  -20.4,  -20.4,  -20.4]
Y["BL"] =     [-6,   -5.9,   -5.9,   -5.8,   -5.6,   -5.4,   -5.3,   -5.1,     -5,   -4.9,   -4.8,   -4.6,   -4.4,   -4.3,   -4.2,   -4.1,   -3.9,   -3.8,   -3.7,   -3.5,   -3.4]
Z["BL"] =   [-1.5,   -5.8,   -9.9,  -14.4,  -19.1,  -23.9,    -31,  -35.7,    -40,  -44.4,  -49.4,  -54.1,  -60.8,  -65.4,  -69.7,  -74.2,  -79.4,  -83.7,  -88.6,  -93.5,  -97.7]
X["BR"] =  [-15.2,    -17,  -17.4,  -17.5,  -17.7,  -17.8,  -17.8,  -17.3,  -17.9,  -17.8,  -17.4,    -17,  -17.3,  -18.6,  -19.9,  -20.2,  -21.3,  -21.5,  -21.9,  -28.4,  -28.2]
Y["BR"] =    [0.2,      1,    1.2,    1.3,    1.7,    2.0,    4.3,    4.5,    5.5,    7.5,    6.9,      7,    2.8,    0.9,    0.8,    0.8,    1.0,    1.2,   -2.2,  -18.9,  -18.9]
Z["BR"] = [-114.6, -118.3, -122.6,   -127, -132.2, -136.5, -144.2, -148.7, -153.1, -159.3, -163.8, -168.5, -175.9, -179.9, -183.7, -187.2, -192.1, -196.6, -201.9, -205.7, -209.7]
X["CL"] =   [28.6,   28.8,   28.3,   28.7,   29.2,   29.5,   30.0,   30.3,   30.5,   30.7,   31.1,   31.2,   30.9,   30.9,   31.2,   31.2,   31.2,   31.2,   31.3,   31.2,   31.3]
Y["CL"] =    [6.3,    6.4,    5.2,    5.3,    5.5,    5.4,    5.3,    5.3,    5.2,    5.1,    5.2,    5.4,      6,    6.1,    5.7,    6.1,    6.1,    6.5,    6.7,    6.8,    6.9]
Z["CL"] =  [-31.1,  -33.1,    -36,  -40.7,  -45.6,  -50.3,  -57.2,  -61.5,    -65,  -68.8,  -76.2,  -80.8,  -87.3,  -91.4,  -95.2, -101.2, -104.8, -110.5, -116.7, -120.5, -124.3]
X["CR"] =    [5.8,    5.7,    5.6,    5.9,    5.9,    6.0,    6.4,    6.5,    6.9,    7.9,    8.3,      8,    8.2,    8.5,    8.4,    4.9,    5.6,    5.8,      6,    6.3,    7.3]
Y["CR"] =    [3.6,    2.6,    1.5,    0.5,   -0.6,   -1.5,   -3.1,   -3.9,   -4.8,   -5.5,   -5.9,   -6.7,   -7.9,     -9,   -9.7,  -11.6,  -12.6,  -13.5,  -14.7,  -15.7,  -16.4]
Z["CR"] =  [-74.9, -77.45,    -82,  -86.8,  -89.7,  -94.3, -102.9, -106.5, -111.6,   -113, -117.6, -124.7, -131.3, -136.8, -140.7, -143.3, -149.6, -153.3, -158.7, -163.2, -168.6]
X["FL"] =  [-13.5,  -13.8,  -13.7,  -13.6,  -14.0,  -13.7,  -13.7,  -13.5,  -13.6,  -13.8,  -14.1,  -13.9,  -14.2,  -14.3,  -14.5,  -14.6,  -14.5,  -14.6,  -14.9,     15,  -15.1]
Y["FL"] =   [20.3,   20.2,   20.5,   20.9,   20.9,   21.4,   21.9,   22.2,   22.4,   22.5,   22.6,   23.1,   23.4,   23.5,   23.7,   23.7,   24.0,   24.2,   24.4,   24.5,   24.5]
Z["FL"] =   [-3.8,   -5.9,   -9.9,  -14.4,  -18.9,  -23.9,    -31,  -35.7,  -40.4,  -44.5,  -49.4,  -54.9,  -62.8,  -67.1,  -71.5,  -75.8,  -80.8,  -85.1,  -90.7,    -95,  -98.7]
X["FR"] =  [-12.1,  -12.5,  -12.9,  -12.9,  -12.9,  -12.6,  -12.7,  -12.5,  -12.6,  -12.5,  -12.6,  -12.6,  -12.9,  -12.9,    -13,  -13.2,  -13.4,  -13.5,  -13.7,  -13.8,  -13.8]
Y["FR"] =    [3.9,    3.9,    2.4,    2.3,      2,    1.9,    1.4,    1.3,      1,    0.9,    0.8,    0.6,    0.4,    0.4,    0.1,      0,      0,   -0.2,   -0.4,   -0.5,   -0.3]
Z["FR"] = [-109.4, -113.8, -117.7, -122.3, -126.9, -132.2, -138.7, -143.6, -147.9, -152.4, -157.2, -161.7, -169.1, -173.4, -177.5, -181.8,   -187, -191.2, -195.6, -200.4, -205.2]
# Below are the measured values after taking off the final weight of ~10kg and shows how the zero value changed probably due to some saturation
# X0["BL"] = -20.4; Y0["BL"] = -6.2; Z0["BL"] = -1.5; X0["BR"] = -19.4; Y0["BR"] = -4.6; Z0["BR"] = -114.5; X0["CL"] = 28.0; Y0["CL"] = 5.5; Z0["CL"] = -29.9; X0["CR"] = 4.1; Y0["CR"] = 3.4; Z0["CR"] = -75.3; X0["FL"] = -14.4; Y0["FL"] = 20.1; Z0["FL"] = -3.8; X0["FR"] = -12.9; Y0["FR"] = 3.5; Z0["FR"] = -110.9
# fmt:on


def get_args() -> argparse.Namespace:
    """Parses CLI arguments"""

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        dest="output",
        help="Path of output figure SVG, e.g., 'ft_slopes.svg'",
    )
    parser.add_argument(
        "--latex",
        "-l",
        action="store_true",
        dest="latex",
        help="Render text in Latex",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    assert hasattr(plt.cm, "rainbow")
    cmap = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
    plt.rcParams["svg.fonttype"] = "none"
    if args.latex:
        plt.rcParams.update(
            {"text.usetex": True, "font.family": "Computer Modern Roman"}
        )

    ALL_FTS = ["FL", "FR", "CL", "CR", "BL", "BR"]
    for fts in ALL_FTS:
        x = [abs(a) - abs(X[fts][0]) for a in X[fts]]
        y = [abs(a) - abs(Y[fts][0]) for a in Y[fts]]
        z = [abs(a) - abs(Z[fts][0]) for a in Z[fts]]
        f = np.sqrt(np.square(x) + np.square(y) + np.square(z))

        f_samples = np.expand_dims(np.array(f), 1)
        W_samples = np.expand_dims(np.array(W), 1)
        model = LinearRegression().fit(W_samples, f_samples)
        offset = model.intercept_
        slope = model.coef_

        print(f"{fts} slope {slope[0][0]}")

        x_plot = np.arange(0, 11, 0.1)
        y_plot = model.predict(np.expand_dims(x_plot, 1))
        c = next(cmap)
        plt.plot(
            W,
            f,
            "o",
            color=c,
            label=f"{fts} measurements",
        )
        plt.plot(
            x_plot, y_plot, color=c, linewidth=0.5, label=f"{fts} slope {slope[0][0]}"
        )
    plt.plot(W, [9.81 * w for w in W], "r", linewidth=1, label="Ideal slope 9.81")
    plt.xlabel("Weight [kg]")
    plt.ylabel("Measured Force [N]")
    plt.xlim([0, 10.2])
    plt.legend()

    if args.output is not None:
        plt.savefig(os.path.expanduser(args.output), format="svg")

    plt.show()
