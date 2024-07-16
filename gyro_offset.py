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

"""Plot MaRTA's pose from CSV."""

__author__ = "Levin Gerdes and Hugo Leblond"


import argparse
import os
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utm  # type: ignore

from euler import *
from export_logs import format_floats


def get_args() -> argparse.Namespace:
    """Parses CLI arguments"""

    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-i",
        type=str,
        default="/media/srl/Nuevo vol/bardenas/2023-07-21_17-45-42/",
        dest="path",
        help="Path of containing input CSVs, e.g. '/path/to/dataset/2023-07-21_17-45-42'",
    )
    parser.add_argument(
        "--precision",
        "-p",
        type=int,
        default=12,
        dest="precision",
        help="Num. of digits behind decimal point in CSV",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_false",
        dest="show_fig",
        help="Don't show figure. E.g. if you want to run this in batch",
    )
    parser.add_argument(
        "--custom-offset",
        "-c",
        type=float,
        default=None,
        dest="custom_offset",
        help="Use this custom heading offset for FOG",
    )

    return parser.parse_args()


def find_closest_timestamp_data(
    short_list: List[Tuple[float, Any]],
    long_list: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """
    Find the closest timestamp in a long list for each timestamp in a short list of tuples.

    Parameters:
    - short_list: The short list of tuples containing timestamps and another value.
    - long_list: The long list of tuples containing timestamps (and another value) to search from.

    Returns:
    - list of tuples: A list of tuples where each tuple contains the closest timestamp from the long list.
    """
    closest_timestamps = []
    copy_long_list = list(long_list)

    for short_tuple in short_list:
        timestamp = short_tuple[0]
        closest_tuple = min(copy_long_list, key=lambda x: abs(x[0] - timestamp))
        closest_timestamps.append(closest_tuple)

    return closest_timestamps


def float_list_from_csv(csv_path: str, column_name: str) -> List[float]:
    """
    Read a CSV file, extract a specified column by name, and parse the values from strings to a list of floats.

    Parameters:
    - csv_name: The path to the CSV file.
    - column_name: The name of the column to extract.

    Returns:
    - list of float: A list of float values extracted from the specified column.
    """
    # import the gnss data
    df = pd.read_csv(csv_path)
    positions_column = df[column_name]
    # Parse positions from strings to lists
    list_csv = [float(pos) for pos in positions_column]
    return list_csv


def compute_tangents(
    points: List[Tuple[float, float]]
) -> Tuple[List[float], List[float]]:
    """
    Compute the tangent (in radians) for each 2D point and optionally apply smoothening.

    Args:
        points: List of 2D points [(x1, y1), (x2, y2), ...].

    Returns:
        avg_tangents: Tangents with a sliding window of size 3
        tangents: Array of tangent angles in radians.
    """
    x = np.array([p[0] for p in points[::10]])
    y = np.array([p[1] for p in points[::10]])

    # Calculate tangent angles for each point
    tangent_angles = np.arctan2(np.diff(y), np.diff(x))
    raw_directions = tangent_angles.tolist()

    # Compute sliding window mean
    df = pd.DataFrame(tangent_angles)
    mean_directions_df = df.rolling(3).mean()
    directions = mean_directions_df[0].to_list()

    # Add last entry again to reach original length
    directions.append(directions[-1])
    raw_directions.append(raw_directions[-1])

    # Replace NANs with zeros
    directions_arr = np.array(directions)
    raw_directions_arr = np.array(raw_directions)
    directions_arr[np.isnan(directions)] = 0
    raw_directions_arr[np.isnan(raw_directions)] = 0

    return (
        directions_arr.tolist(),
        raw_directions_arr.tolist(),
    )


def gnss_to_utm(
    gnss_pos: List[Tuple[float, float, float]]
) -> List[Tuple[float, float]]:
    """
    Convert GNSS latitude and longitude coordinates to UTM coordinates.

    Parameters:
    - gnss_pos: A list of tuples containing (latitude, longitude, altitude).

    Returns:
    - list: UTM coordinates as (easting, northing).
    """
    lat_gnss_coordinates = [pos[0] for pos in gnss_pos]
    long_gnss_coordinates = [pos[1] for pos in gnss_pos]

    utm_coordinates: List[Tuple[float, float]] = []
    for lat, lon in zip(lat_gnss_coordinates, long_gnss_coordinates):
        utm_easting, utm_northing, _, _ = utm.from_latlon(
            lat, lon, force_zone_number=30, force_zone_letter="T"
        )
        # print(utm_easting, utm_northing)
        utm_coordinates.append((utm_easting, utm_northing))

    return utm_coordinates


def average_gnss_positions(
    gnss_pos: List[Tuple[float, float]], window: int
) -> List[Tuple[float, float]]:
    """
    Computes sliding window average GNSS positions
    """
    avgs = []
    gnss = np.array(gnss_pos)
    for i in range(len(gnss)):
        lower = max(i - window // 2, 0)
        upper = min(i + window // 2, len(gnss))
        avgs.append(np.mean(gnss[lower:upper], 0))
    return avgs


if __name__ == "__main__":
    args = get_args()
    plt.rcParams["svg.fonttype"] = "none"

    utm_easting = float_list_from_csv(
        os.path.join(args.path, "GNSS.csv"), "UTM_Easting"
    )
    utm_northing = float_list_from_csv(
        os.path.join(args.path, "GNSS.csv"), "UTM_Northing"
    )
    gnss_t_raw = float_list_from_csv(os.path.join(args.path, "GNSS.csv"), "Timestamp")
    gnss_positions_raw = list(zip(utm_easting, utm_northing))
    gnss_positions = average_gnss_positions(gnss_positions_raw, 16)
    gnss_t = gnss_t_raw[::10]

    fog_heading = float_list_from_csv(os.path.join(args.path, "FOG.csv"), "Angle_Z")
    fog_t = float_list_from_csv(os.path.join(args.path, "FOG.csv"), "Timestamp")
    imu_x = float_list_from_csv(os.path.join(args.path, "IMU.csv"), "Orientation_X")
    imu_y = float_list_from_csv(os.path.join(args.path, "IMU.csv"), "Orientation_Y")
    imu_z = float_list_from_csv(os.path.join(args.path, "IMU.csv"), "Orientation_Z")
    imu_w = float_list_from_csv(os.path.join(args.path, "IMU.csv"), "Orientation_W")
    imu_t = float_list_from_csv(os.path.join(args.path, "IMU.csv"), "Timestamp")
    imu = zip(imu_x, imu_y, imu_z, imu_w)
    imu_heading = [
        quaternion_to_euler(Quaternion(q[0], q[1], q[2], q[3])).z + np.pi for q in imu
    ]
    imu_heading = [wrap_angle(x) for x in imu_heading]
    gnss_tangent, gnss_tangent_raw = compute_tangents(gnss_positions)

    # check part near the end
    imu_indices = range(int(8 / 10 * len(imu_heading)), int(9 / 10 * len(imu_heading)))
    common_time = [imu_t[imu_indices[0]], imu_t[imu_indices[-1]]]
    fog_indices = [
        i
        for i, timestamp in enumerate(fog_t)
        if common_time[0] <= timestamp <= common_time[1]
    ]
    imu_compare = [imu_heading[i] for i in imu_indices]
    fog_compare = [fog_heading[i] for i in fog_indices]

    fog_raw = fog_heading

    if args.custom_offset is None:
        fog_offset = (np.mean(np.unwrap(imu_compare))) - np.mean(np.unwrap(fog_compare))
        print(f"{args.path} fog offset {fog_offset}")
    else:
        fog_offset = args.custom_offset
        print(f"{args.path} custom fog offset {fog_offset}")

    fog_heading = [wrap_angle(h + fog_offset) for h in fog_heading]

    plt.subplot(2, 1, 1)
    plt.plot(
        gnss_t,
        gnss_tangent,
        color="purple",
        label="Sliding window GNSS heading",
    )
    plt.plot(imu_t, imu_heading, color="blue", label="IMU heading")
    plt.plot(fog_t, fog_raw, label="Raw FOG heading")
    plt.plot(fog_t, fog_heading, color="red", label="Corrected FOG heading")
    plt.xlabel("Time [s]")
    plt.ylabel("Heading [rad]")
    plt.legend()

    all_x = [pos[0] for pos in gnss_positions_raw]
    all_y = [pos[1] for pos in gnss_positions_raw]

    plt.subplot(2, 1, 2)
    plt.scatter(all_x, all_y, s=20, label="GNSS positions")

    heading_step = len(fog_heading) // len(gnss_positions_raw)
    heading_matches = find_closest_timestamp_data(
        list(zip(gnss_t_raw, gnss_positions_raw)), list(zip(fog_t, fog_heading))
    )
    heading_match_values = [x[1] for x in heading_matches]
    is_first_arrow = True
    for point, orientation in zip(gnss_positions_raw, heading_match_values):
        x, y = point[0], point[1]
        dx = 0.5 * np.cos(orientation)
        dy = 0.5 * np.sin(orientation)

        plt.arrow(
            x,
            y,
            dx,
            dy,
            head_width=0.01,
            head_length=0.01,
            fc="red",
            ec="red",
            label="Corrected FOG Heading" if is_first_arrow else "_nolegend_",
        )
        is_first_arrow = False

    # Write corrected FOG values
    df = pd.read_csv(os.path.join(args.path, "FOG.csv"))
    quaternions = [euler_to_quaternion(Euler(0, 0, z)) for z in fog_heading]
    for q in quaternions:
        assert q.x == 0
        assert q.y == 0
    df["Orientation_Z"] = format_floats([q.z for q in quaternions], args.precision)
    df["Orientation_W"] = format_floats([q.w for q in quaternions], args.precision)
    df["Angle_Z"] = fog_heading
    df.to_csv(os.path.join(args.path, "FOG_CORRECTED.csv"), index=False)

    plt.xlabel("UTM Easting [m]")
    plt.ylabel("UTM Northing [m]")
    # plt.gca().set_aspect("equal")
    plt.legend()

    plt.savefig(
        os.path.join(args.path, "fog_offset_correction.svg"),
        format="svg",
    )

    if args.show_fig:
        plt.show()
