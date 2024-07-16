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
Find offsets of the rear bogie

The script reads the dynamic transformations (TF.csv) and writes a new file with
offset-corrected rear bogie values (TF_CORRECTED_REAR_BOGIE.csv).

Offsets are calculated based on start and end bogie calibration workaround
(aka. bogie dance), during which the rover is lifted up at the rear bogie
(which can be seen in the rover pitch (IMU)) and the rear bogie is subsequently
moved to both extremes. The offset will be between those extrema.

Because of the bogie dance, parts of the traverse are 'invalid'.
The new transformations file thus only contains the transformations that were
recorded in-between dances.

To see all options, please invoke
bogie_offset.py --help
"""

__author__ = "Levin Gerdes"


import argparse
import csv
import datetime as dt
import os
from typing import List, Tuple

import matplotlib.dates as md
import matplotlib.pyplot as plt
import numpy as np
import progressbar  # type: ignore

from euler import (
    Euler,
    Quaternion,
    euler_to_quaternion,
    quaternion_to_euler,
    wrap_angle,
    wrap_euler,
)


def to_datenum(timestamps: List[float]) -> List[float]:
    dates = [dt.datetime.fromtimestamp(ts) for ts in timestamps]
    return md.date2num(dates)


def get_args() -> argparse.Namespace:
    """Parses CLI arguments"""

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        default="~/data/bardenas2023/dataset/2023-07-21_17-45-42",
        dest="path",
        help="Path containing input CSVs, e.g. '/path/to/dataset/2023-07-21_17-45-42'",
    )
    parser.add_argument(
        "--latex",
        "-l",
        action="store_true",
        dest="latex",
        help="Render text in Latex",
    )
    parser.add_argument(
        "--plot-all-bogies",
        "-pab",
        action="store_true",
        dest="plot_all_bogies",
        help="Plot all bogies, not just the rear",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_false",
        dest="show_fig",
        help="Don't show figure. E.g. if you want to run this in batch",
    )
    parser.add_argument(
        "--fig-name",
        "-fn",
        type=str,
        default="bogie_offset_corrections.svg",
        dest="fig_name",
        help="Name of the output image",
    )
    parser.add_argument(
        "--fig-w",
        "-fw",
        type=float,
        default=19.20,
        dest="fig_w",
        help="Output image width",
    )
    parser.add_argument(
        "--fig-h",
        "-fh",
        type=float,
        default=10.80,
        dest="fig_h",
        help="Output image height",
    )

    return parser.parse_args()


def correct_tf(
    start_offset: float,
    end_offset: float,
    traverse_start: float,
    traverse_end: float,
    start_time: float,
    end_time: float,
    num_tf_messages: int,
    path: str,
) -> Tuple[List[float], List[float]]:
    # Write new output file with offset values
    path_tf = os.path.join(path, "TF.csv")
    path_tf_corrected = os.path.join(path, "TF_CORRECTED_REAR_BOGIE.csv")
    plot_data: Tuple[List[float], List[float]] = ([], [])
    with open(path_tf, "r") as f_orig, open(path_tf_corrected, "w") as f_corrected:
        reader = csv.DictReader(f_orig)
        writer = csv.DictWriter(
            f_corrected,
            fieldnames=[
                "Timestamp",
                "Frame_ID",
                "Child_Frame_ID",
                "TX",
                "TY",
                "TZ",
                "QX",
                "QY",
                "QZ",
                "QW",
            ],
            dialect="unix",
            quoting=csv.QUOTE_NONE,
        )
        bar = progressbar.ProgressBar(maxval=num_tf_messages).start()
        for i, row in enumerate(reader):
            # skip messages outside 'valid' traverse
            timestamp = float(row["Timestamp"]) / 1e9
            if timestamp < traverse_start or traverse_end < timestamp:
                continue

            if row["Child_Frame_ID"] == "link_bogie_MRB":
                offset_slope = (end_offset - start_offset) / (end_time - start_time)
                offset = start_offset + offset_slope * (timestamp - start_time)

                q_orig = Quaternion(
                    x=float(row["QX"]),
                    y=float(row["QY"]),
                    z=float(row["QZ"]),
                    w=float(row["QW"]),
                )

                e = quaternion_to_euler(q_orig)
                e.y = wrap_angle(e.y - offset)
                q_corrected = euler_to_quaternion(e)

                new_row = row
                new_row["QX"] = q_corrected.x
                new_row["QY"] = q_corrected.y
                new_row["QZ"] = q_corrected.z
                new_row["QW"] = q_corrected.w

                writer.writerow(new_row)

                plot_data[0].append(timestamp)
                plot_data[1].append(wrap_euler(quaternion_to_euler(q_corrected)).y)

            else:
                writer.writerow(row)
            bar.update(i)

    return plot_data


if __name__ == "__main__":
    args = get_args()

    print(f"Path: {args.path}")

    plt.figure(figsize=(args.fig_w, args.fig_h))
    # keep text editable in svg and don't embed font
    plt.rcParams["svg.fonttype"] = "none"

    if args.latex:
        # requires the following packages:
        # sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
        plt.rcParams.update(
            {"text.usetex": True, "font.family": "Computer Modern Roman"}
        )

    # format x axis labels
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    xfmt = md.DateFormatter("%Y-%m-%d %H:%M:%S")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(xfmt)

    timestamps_rear: List[float] = []
    euler_angles_rear: List[Euler] = []
    timestamps_left: List[float] = []
    euler_angles_left: List[Euler] = []
    timestamps_right: List[float] = []
    euler_angles_right: List[Euler] = []
    path_tf = os.path.join(args.path, "TF.csv")
    num_tf_messages: int = 0
    with open(path_tf, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            num_tf_messages += 1
            if row["Child_Frame_ID"] == "link_bogie_MRB":
                q = Quaternion(
                    x=float(row["QX"]),
                    y=float(row["QY"]),
                    z=float(row["QZ"]),
                    w=float(row["QW"]),
                )
                timestamps_rear.append(float(int(row["Timestamp"]) / 1e9))
                euler_angles_rear.append(wrap_euler(quaternion_to_euler(q)))
            if row["Child_Frame_ID"] == "link_bogie_LFB":
                q = Quaternion(
                    x=float(row["QX"]),
                    y=float(row["QY"]),
                    z=float(row["QZ"]),
                    w=float(row["QW"]),
                )
                timestamps_left.append(float(int(row["Timestamp"]) / 1e9))
                euler_angles_left.append(wrap_euler(quaternion_to_euler(q)))
            if row["Child_Frame_ID"] == "link_bogie_RFB":
                q = Quaternion(
                    x=float(row["QX"]),
                    y=float(row["QY"]),
                    z=float(row["QZ"]),
                    w=float(row["QW"]),
                )
                timestamps_right.append(float(int(row["Timestamp"]) / 1e9))
                euler_angles_right.append(wrap_euler(quaternion_to_euler(q)))

    timestamps_imu: List[float] = []
    euler_angles_imu: List[Euler] = []
    path_imu = os.path.join(args.path, "IMU.csv")
    with open(path_imu, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = Quaternion(
                x=float(row["Orientation_X"]),
                y=float(row["Orientation_Y"]),
                z=float(row["Orientation_Z"]),
                w=float(row["Orientation_W"]),
            )
            timestamps_imu.append(float(int(row["Timestamp"]) / 1e9))
            euler_angles_imu.append(wrap_euler(quaternion_to_euler(q)))

    # plot all imu angles
    ypoints_imu_pitch = [e.y for e in euler_angles_imu]
    plt.plot(
        to_datenum(timestamps_imu),
        ypoints_imu_pitch,
        "y",
        label="IMU pitch",
        linewidth=1,
    )

    # plot bogie angles
    ypoints_rear = [e.y for e in euler_angles_rear]
    plt.plot(
        to_datenum(timestamps_rear),
        ypoints_rear,
        color="orange",
        label="Rear bogie recorded",
        linewidth=1,
    )

    if args.plot_all_bogies:
        ypoints_left = [e.y for e in euler_angles_left]
        plt.plot(
            to_datenum(timestamps_left),
            ypoints_left,
            "--",
            label="Left bogie",
            linewidth=1,
        )
        ypoints_right = [e.y for e in euler_angles_right]
        plt.plot(
            to_datenum(timestamps_right),
            ypoints_right,
            "--",
            label="Right bogie",
            linewidth=1,
        )

    # min/max indices
    # Only check within first and last ten percent of a traverse to avoid
    # a 'drifted' bogie reading later on in the traverse to be counted as
    # a min/max value.
    val_with_time = list(zip(timestamps_rear, [e.y for e in euler_angles_rear]))
    first_ten_percent = len(val_with_time) // 10
    last_ten_percent = 9 * len(val_with_time) // 10
    start_min_index = np.argmin(val_with_time[:first_ten_percent], axis=0)
    start_max_index = np.argmax(val_with_time[:first_ten_percent], axis=0)
    end_min_index = np.argmin(val_with_time[last_ten_percent:], axis=0)
    end_max_index = np.argmax(val_with_time[last_ten_percent:], axis=0)

    # min/max values
    start_min = val_with_time[start_min_index[1]]
    start_max = val_with_time[start_max_index[1]]
    end_min = val_with_time[last_ten_percent + end_min_index[1]]
    end_max = val_with_time[last_ten_percent + end_max_index[1]]

    # check whether min/max are more than 0.5 rad apart
    has_start_dance: bool = 0.5 < start_max[1] - start_min[1]
    has_end_dance: bool = 0.5 < end_max[1] - end_min[1]

    if has_start_dance:
        plt.plot(to_datenum([start_min[0]]), start_min[1], "gv")
        plt.plot(to_datenum([start_max[0]]), start_max[1], "g^")
    if has_end_dance:
        plt.plot(to_datenum([end_min[0]]), end_min[1], "rv")
        plt.plot(to_datenum([end_max[0]]), end_max[1], "r^")

    plt.xlim(to_datenum([timestamps_rear[0], timestamps_rear[-1]]))
    plt.ylim([0.6, 1.65])
    plt.ylim(
        [
            min(min(ypoints_imu_pitch), start_min[1], end_min[1]) - 0.05,
            max(max(ypoints_imu_pitch), start_max[1], end_max[1]) + 0.05,
        ]
    )
    # plt.ylim([min(start_min[1], end_min[1])-0.1, max(start_max[1], end_max[1])])

    # offsets
    # pull everything closer to 0 even if there's no start dance
    start_offset = (
        (start_max[1] + start_min[1]) / 2.0 if has_start_dance else val_with_time[0][1]
    )
    end_offset = (end_max[1] + end_min[1]) / 2.0 if has_end_dance else start_offset

    print(f"Offsets: {start_offset} at start and {end_offset} the end")

    # offset line
    start_time: float = (
        (start_max[0] + start_min[0]) / 2.0 if has_start_dance else timestamps_rear[0]
    )
    end_time: float = (
        (end_max[0] + end_min[0]) / 2.0 if has_end_dance else timestamps_rear[-1]
    )
    plt.plot(
        to_datenum([start_time, end_time]),
        [start_offset, end_offset],
        "b:",
        label="Offset",
    )
    plt.axhspan(
        start_offset,
        end_offset,
        facecolor="b",
        alpha=0.05,
        label="Offset range",
    )

    # mark valid region (in-between bogie dances)
    traverse_start = start_time + 10 if has_start_dance else float(timestamps_rear[0])
    value = dt.datetime.fromtimestamp(int(traverse_start))
    traverse_end = end_time - 10 if has_end_dance else float(timestamps_rear[-1])
    print(
        f"Rec. start:\t{timestamps_rear[0]} → {dt.datetime.fromtimestamp(timestamps_rear[0])}"
    )
    print(
        f"Rec. end:\t{timestamps_rear[-1]} → {dt.datetime.fromtimestamp(timestamps_rear[-1])}"
    )
    print(
        f"Traverse start:\t{traverse_start} → {dt.datetime.fromtimestamp(traverse_start)}"
    )
    print(f"Traverse end:\t{traverse_end} → {dt.datetime.fromtimestamp(traverse_end)}")
    plt.axvspan(
        to_datenum([traverse_start])[0],
        to_datenum([traverse_end])[0],
        facecolor="g",
        alpha=0.05,
        label="Traverse",
    )

    corrected_plot_data = correct_tf(
        start_offset,
        end_offset,
        traverse_start,
        traverse_end,
        start_time,
        end_time,
        num_tf_messages,
        args.path,
    )

    plt.plot(
        to_datenum(corrected_plot_data[0]),
        corrected_plot_data[1],
        color="blue",
        label="Rear bogie corrected",
        linewidth=1,
    )

    plt.title(os.path.basename(os.path.normpath(args.path)))
    plt.xlabel("Time")
    plt.ylabel("Angle [rad]")
    plt.legend()

    plt.savefig(
        os.path.join(args.path, args.fig_name),
        format="svg",
    )

    if args.show_fig:
        plt.show()
