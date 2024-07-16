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
Extracts mcap rosbags.

This script scans the mcap bags and extracts the messages and timestamps.
"""

__author__ = "Levin Gerdes and Hugo Leblond"


import argparse
import csv
import os
from enum import Enum, auto
from typing import Any, Callable, Dict, List, TextIO, Tuple

import cv2 as cv2
import imageio
import numpy as np
import progressbar  # type: ignore
import utm  # type: ignore
from rosbags.interfaces import Connection
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from rosbags.typesys import get_types_from_msg, register_types

from euler import Quaternion, quaternion_to_euler, wrap_angle

# The OptrisP640 thermal camera has a resolution of 640*480=307200 pixels
# https://mesurex.com/catalogo/productos/camaras-termograficas/optris-pi-640/
OPTRIS_PI640_PIXELS = 640 * 480
RS_DEPTH_PIXELS = 848 * 480

# Registering new TemperatureMatrix custom message
STRIDX_MSG = """
std_msgs/Header header
uint32 height
uint32 width
float32[] data
"""
register_types(get_types_from_msg(STRIDX_MSG, "thermal_camera/msg/TemperatureMatrix"))


class Sensors(Enum):
    THERMAL = auto()
    THERMAL_RGB = auto()
    FTS_FL = auto()
    FTS_FR = auto()
    FTS_CL = auto()
    FTS_CR = auto()
    FTS_BL = auto()
    FTS_BR = auto()
    RS_DEPTH = auto()
    RS_COLOR = auto()
    XB3_LEFT = auto()
    XB3_RIGHT = auto()
    IMU = auto()
    FOG = auto()
    GNSS = auto()
    TF = auto()
    TF_STATIC = auto()


def get_args() -> argparse.Namespace:
    """Parses CLI arguments"""

    parser = argparse.ArgumentParser(description="")

    # fmt: off
    parser.add_argument("--input-path",  "-i",    type=str,   default="~/data/bardenas2023/dataset",   dest="input_path",    help="Parent directory of input mcaps. E.g. '/path/to/recordings' with subdirs '2023-07-21_17-45-42' etc.")
    parser.add_argument("--output-path", "-o",    type=str,   default="~/data/bardenas2023/extracted", dest="output_path",   help="Output will be written here")
    parser.add_argument("--compression", "-c",    type=int,   default=9,                               dest="png_comp_rate", help="PNG compression [0..9], lowest to highest rate")
    parser.add_argument("--precision",   "-p",    type=int,   default=12,                              dest="precision",     help="Num. of digits behind decimal point in CSV")
    parser.add_argument("--max-temp",    "-tmax", type=float, default=50,                              dest="temp_max",      help="Maximum temperature. Only used for export of thermal float matrices to PNG. Maximum temperature and above = White")
    parser.add_argument("--min-temp",    "-tmin", type=float, default=10,                              dest="temp_min",      help="Minimum temperature. Only used for export of thermal float matrices to PNG. Minimum temperature and below = Black")
    # fmt: on

    return parser.parse_args()


def crop_grayscale_image(
    image: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """
    Crop a grayscale image to a specified target height and width,
    centered around the original image's center.

    Parameters:
    - image (numpy.ndarray): The input grayscale image.
    - target_height (int): The desired height of the cropped image.
    - target_width (int): The desired width of the cropped image.

    Returns:
    - numpy.ndarray: The cropped grayscale image.
    """
    # Get the height and width of the original image
    height, width = image.shape

    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2

    # Calculate the starting and ending coordinates for cropping
    start_x = center_x - target_width // 2
    end_x = start_x + target_width
    start_y = center_y - target_height // 2
    end_y = start_y + target_height

    # Crop the image
    cropped_image = image[start_y:end_y, start_x:end_x]

    return cropped_image


def gnss_to_utm(gnss_pos: Tuple[float, float, float]) -> List[float]:
    """
    Convert GNSS latitude, longitude, and altitude coordinates to UTM coordinates.
    Returns the input altitude as output altitude.

    Parameters:
    - gnss_pos: [latitude, longitude, altitude].

    Returns:
    - list: UTM coordinates in the format [easting, northing, altitude].
    """
    lat_gnss_coordinates = gnss_pos[0]
    long_gnss_coordinates = gnss_pos[1]

    utm_easting, utm_northing, _, _ = utm.from_latlon(
        lat_gnss_coordinates,
        long_gnss_coordinates,
        force_zone_number=30,
        force_zone_letter="T",
    )

    return [utm_easting, utm_northing, gnss_pos[2]]


def format_floats(input: List[Any], precision: int = 12) -> List[Any]:
    """
    Use positional instead of scientific notation for all floats in input array.
    Values that are not floats are left unchanged.
    """
    return [
        (
            np.format_float_positional(x, precision, trim="-")
            if isinstance(x, float)
            else x
        )
        for x in input
    ]


def export_rgb_img(msg, _0, path: str, _1, _2) -> None:
    img = np.frombuffer(msg.data, dtype=np.uint8)
    img = img.reshape((msg.height, msg.width, -1))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, args.png_comp_rate])


def export_xb3(msg, _0, path: str, _1, _2) -> None:
    img = np.frombuffer(msg.data, dtype=np.uint8)
    img = img.reshape((msg.height, msg.width))
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GBRG2BGR)
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, args.png_comp_rate])


def export_rs_depth(
    msg, timestamp, path: str, writer: csv.DictWriter, keys: List[str]
) -> None:
    depth = np.frombuffer(msg.data, dtype=np.uint16)
    depth = np.minimum(np.maximum(depth, 0), 65635)  # limit to [0,max16bit]
    depth = np.reshape(depth, (msg.height, msg.width))

    # Save depth as 24-bit RGB PNG.
    # Blue channel has the lowest and red the highest significance.
    # Incoming depth is 16-bit, so red is never used.
    img = np.zeros((msg.height, msg.width, 3), np.uint8)
    img[:, :, 0] = depth & 0x0000FF
    img[:, :, 1] = (depth & 0x00FF00) >> 8
    img[:, :, 2] = (depth & 0xFF0000) >> 16
    cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, args.png_comp_rate])

    # Save depth as 16-bit grayscale PNG.
    img_16bit = np.zeros((msg.height, msg.width, 1), np.uint16)
    img_16bit = depth
    imageio.imsave(
        os.path.dirname(path) + "_16bit/" + os.path.basename(path), img_16bit
    )

    # Save depth in CSV
    val = [timestamp]
    val.extend(np.frombuffer(msg.data, dtype=np.uint16))
    writer.writerow(dict(zip(keys, format_floats(val, args.precision))))


def export_thermal_float(
    msg, timestamp, path: str, writer: csv.DictWriter, keys: List[str]
) -> None:
    img = np.frombuffer(msg.data[:OPTRIS_PI640_PIXELS], dtype=np.float32)
    img = img.reshape((msg.height, msg.width))
    img = np.interp(img, (args.temp_min, args.temp_max), (0, 255)).astype(np.float32)
    cv2.imwrite(path, cv2.Mat(img), [cv2.IMWRITE_PNG_COMPRESSION, args.png_comp_rate])

    val = [timestamp]
    # Rounding because the Optris P640 reports a sensitivity of 75mK, ca. 0.07°C
    val.extend([round(x, 2) for x in msg.data[:OPTRIS_PI640_PIXELS]])

    # Write individual CSV files per image instead of one large file
    # csv_path = pathlib.Path(path).with_suffix(".csv")
    # with open(csv_path, "w") as c:
    #     small_writer = csv.DictWriter(c, fieldnames=keys)
    #     # small_writer.writeheader()
    #     small_writer.writerow(dict(zip(keys, val)))

    writer.writerow(dict(zip(keys, format_floats(val, args.precision))))


def export_fts(msg, timestamp, _, writer: csv.DictWriter, keys: List[str]) -> None:
    # https://docs.ros2.org/latest/api/geometry_msgs/msg/Wrench.html
    val = [
        timestamp,
        msg.wrench.force.x,
        msg.wrench.force.y,
        msg.wrench.force.z,
        msg.wrench.torque.x,
        msg.wrench.torque.y,
        msg.wrench.torque.z,
    ]

    writer.writerow(dict(zip(keys, format_floats(val, args.precision))))


def export_imu(msg, timestamp, _, writer: csv.DictWriter, keys: List[str]) -> None:
    """
    Read and export IMU messages.

    The IMU does not provide covariances (i.e., all covariance matrices only contain 0s),
    so we do not export them.

    Messages are of type IMU:
    https://docs.ros2.org/latest/api/sensor_msgs/msg/Imu.html
    """
    val = [
        timestamp,
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w,
        msg.angular_velocity.x,
        msg.angular_velocity.y,
        msg.angular_velocity.z,
        msg.linear_acceleration.x,
        msg.linear_acceleration.y,
        msg.linear_acceleration.z,
    ]

    e = quaternion_to_euler(
        Quaternion(
            x=msg.orientation.x,
            y=msg.orientation.y,
            z=msg.orientation.z,
            w=msg.orientation.w,
        )
    )
    val.extend([wrap_angle(e.x), wrap_angle(e.y), wrap_angle(e.z)])

    writer.writerow(dict(zip(keys, format_floats(val, args.precision))))


def export_fog(msg, timestamp, _, writer: csv.DictWriter, keys: List[str]) -> None:
    """
    Read and export Fibre Optic Gyro messages.

    The Fibre Optic Gyro does not provide covariances nor linear accelearations,
    and it only measures rotations around its z axis.
    We only use the sensor for heading estimations.

    Messages are of type IMU:
    https://docs.ros2.org/latest/api/sensor_msgs/msg/Imu.html
    """
    e = quaternion_to_euler(
        Quaternion(
            x=msg.orientation.x,
            y=msg.orientation.y,
            z=msg.orientation.z,
            w=msg.orientation.w,
        )
    )

    val = [
        timestamp,
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w,
        msg.angular_velocity.z,
        wrap_angle(e.z),
    ]

    writer.writerow(dict(zip(keys, format_floats(val, args.precision))))


def export_gnss(msg, timestamp, _, writer: csv.DictWriter, keys: List[str]) -> None:
    # https://docs.ros2.org/latest/api/sensor_msgs/msg/NavSatFix.html
    utm_east, utm_north, _ = gnss_to_utm((msg.latitude, msg.longitude, msg.altitude))

    val = [
        timestamp,
        msg.status.status,
        msg.status.service,
        msg.latitude,
        msg.longitude,
        msg.altitude,
        msg.position_covariance_type,
    ]
    val.extend([round(c, 12) for c in msg.position_covariance])
    val.extend([utm_east, utm_north])

    writer.writerow(dict(zip(keys, format_floats(val, args.precision))))


def export_tf(msg, _0, _1, writer: csv.DictWriter, keys: List[str]) -> None:
    # https://docs.ros2.org/foxy/api/tf2_msgs/msg/TFMessage.html

    for t in msg.transforms:
        val = [
            # Timestamps are part of the individual stamped transforms,
            # not of the overall tf message.
            int(str(t.header.stamp.sec) + str(t.header.stamp.nanosec).zfill(9)),
            t.header.frame_id,
            t.child_frame_id,
            t.transform.translation.x,
            t.transform.translation.y,
            t.transform.translation.z,
            t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w,
        ]

        writer.writerow(dict(zip(keys, format_floats(val, args.precision))))


def get_message_and_timestamp(
    connection: Connection, rawdata: bytes
) -> tuple[Any, int | None]:
    """Returns the deserialized message and its timestamp"""
    msg = deserialize_cdr(rawdata, connection.msgtype)
    timestamp = None
    if hasattr(msg, "header"):
        timestamp = int(
            str(msg.header.stamp.sec) + str(msg.header.stamp.nanosec).zfill(9)
        )
    return msg, timestamp


if __name__ == "__main__":
    args = get_args()

    INPUT_PATH: str = os.path.expanduser(args.input_path)
    OUTPUT_PATH: str = os.path.expanduser(args.output_path)
    print(f"Input path: {INPUT_PATH}\nOutput path: {OUTPUT_PATH}")

    # Get names of subdirectories in INPUT_PATH
    # E.g. "2023-07-20_18-25-21"
    SUBDIRS: List[str] = [
        directory
        for directory in os.listdir(INPUT_PATH)
        if os.path.isdir(os.path.join(INPUT_PATH, directory))
    ]

    # Map topic names to Sensor enum and function that can read and export it
    topic_map: Dict[str, Tuple[Sensors, Callable]] = {
        "/throttle/camera/depth/image_rect_raw": (Sensors.RS_DEPTH, export_rs_depth),
        "/throttle/camera/color/image_raw": (Sensors.RS_COLOR, export_rgb_img),
        "/throttle/thermal_float": (Sensors.THERMAL, export_thermal_float),
        "/throttle/thermal_RGB": (Sensors.THERMAL_RGB, export_rgb_img),
        "/throttle/nav_cam/left/image_raw": (Sensors.XB3_LEFT, export_xb3),
        "/throttle/nav_cam/right/image_raw": (Sensors.XB3_RIGHT, export_xb3),
        "/fts_readings/FTS_FL": (Sensors.FTS_FL, export_fts),
        "/fts_readings/FTS_FR": (Sensors.FTS_FR, export_fts),
        "/fts_readings/FTS_CL": (Sensors.FTS_CL, export_fts),
        "/fts_readings/FTS_CR": (Sensors.FTS_CR, export_fts),
        "/fts_readings/FTS_BL": (Sensors.FTS_BL, export_fts),
        "/fts_readings/FTS_BR": (Sensors.FTS_BR, export_fts),
        "/fog/rotation": (Sensors.FOG, export_fog),
        "/imu/data": (Sensors.IMU, export_imu),
        "/gnss": (Sensors.GNSS, export_gnss),
        "/tf": (Sensors.TF, export_tf),
        "/tf_static": (Sensors.TF_STATIC, export_tf),
    }

    # Map from sensor enum to header entries / column names for the corresponding CSV
    header_map: Dict[Sensors, List[str]] = {}

    header_map[Sensors.FTS_FL] = header_map[Sensors.FTS_FR] = [
        "Timestamp",
        "Force_X",
        "Force_Y",
        "Force_Z",
        "Torque_X",
        "Torque_Y",
        "Torque_Z",
    ]
    header_map[Sensors.FTS_CL] = header_map[Sensors.FTS_CR] = header_map[Sensors.FTS_FL]
    header_map[Sensors.FTS_BL] = header_map[Sensors.FTS_BR] = header_map[Sensors.FTS_FL]

    header_map[Sensors.GNSS] = [
        "Timestamp",
        "Status",
        "Service",
        "Latitude",
        "Longitude",
        "Altitude",
        "Position_Covariance_Type",
        "Position_Covariance_0",
        "Position_Covariance_1",
        "Position_Covariance_2",
        "Position_Covariance_3",
        "Position_Covariance_4",
        "Position_Covariance_5",
        "Position_Covariance_6",
        "Position_Covariance_7",
        "Position_Covariance_8",
        "UTM_Easting",
        "UTM_Northing",
    ]
    header_map[Sensors.FOG] = [
        "Timestamp",
        "Orientation_X",
        "Orientation_Y",
        "Orientation_Z",
        "Orientation_W",
        "Angular_Velocity_Z",
        "Angle_Z",
    ]
    header_map[Sensors.IMU] = [
        "Timestamp",
        "Orientation_X",
        "Orientation_Y",
        "Orientation_Z",
        "Orientation_W",
        "Angular_Velocity_X",
        "Angular_Velocity_Y",
        "Angular_Velocity_Z",
        "Linear_Acceleration_X",
        "Linear_Acceleration_Y",
        "Linear_Acceleration_Z",
        "Angle_X",
        "Angle_Y",
        "Angle_Z",
    ]
    header_map[Sensors.THERMAL] = ["Timestamp"]
    header_map[Sensors.THERMAL].extend(
        [f"Data_{i}" for i in range(OPTRIS_PI640_PIXELS)]
    )
    header_map[Sensors.TF] = [
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
    ]
    header_map[Sensors.RS_DEPTH] = ["Timestamp"]
    header_map[Sensors.RS_DEPTH].extend([f"Data_{i}" for i in range(RS_DEPTH_PIXELS)])
    header_map[Sensors.TF_STATIC] = header_map[Sensors.TF]
    header_map[Sensors.RS_COLOR] = []
    header_map[Sensors.XB3_LEFT] = header_map[Sensors.XB3_RIGHT] = []
    header_map[Sensors.THERMAL_RGB] = []

    for directory in SUBDIRS:
        # Setup export directories
        abs_out_dir: str = os.path.join(OUTPUT_PATH, directory)
        for s in [
            Sensors.RS_COLOR,
            Sensors.RS_DEPTH,
            Sensors.THERMAL,
            Sensors.THERMAL_RGB,
            Sensors.XB3_LEFT,
            Sensors.XB3_RIGHT,
        ]:
            os.makedirs(name=os.path.join(abs_out_dir, s.name), exist_ok=True)
        os.makedirs(
            name=os.path.join(abs_out_dir, Sensors.RS_DEPTH.name + "_16bit"),
            exist_ok=True,
        )

        # Setup CSV Writers with paths to current export directory
        csvs: Dict[Sensors, TextIO] = {}
        csv_writers: Dict[Sensors, csv.DictWriter] = {}
        for s in Sensors:
            if s in [
                Sensors.FTS_FL,
                Sensors.FTS_FR,
                Sensors.FTS_CL,
                Sensors.FTS_CR,
                Sensors.FTS_BL,
                Sensors.FTS_BR,
                Sensors.GNSS,
                Sensors.FOG,
                Sensors.IMU,
                Sensors.THERMAL,
                Sensors.RS_DEPTH,
                Sensors.TF,
                Sensors.TF_STATIC,
            ]:
                csvs[s] = open(os.path.join(abs_out_dir, f"{s.name}.csv"), "w")
                csv_writers[s] = csv.DictWriter(
                    csvs[s],
                    fieldnames=header_map[s],
                    dialect="unix",
                    quoting=csv.QUOTE_NONE,
                )
                csv_writers[s].writeheader()
            else:
                csv_writers[s] = None

        # Read and export mcap in current directory
        with Reader(os.path.join(INPUT_PATH, directory)) as reader:
            print(directory)

            num_messages = sum(c.msgcount for c in reader.connections)

            bar = progressbar.ProgressBar(maxval=num_messages)
            bar.start()
            progress = 0
            for connection, timestamp, rawdata in reader.messages():
                if connection.topic in topic_map.keys():
                    s: Sensors = topic_map[connection.topic][0]
                    f: Callable = topic_map[connection.topic][1]
                    msg, time = get_message_and_timestamp(connection, rawdata)
                    img_path = os.path.join(abs_out_dir, s.name, f"{time}_{s.name}.png")
                    f(msg, time, img_path, csv_writers[s], header_map[s])
                progress += 1
                bar.update(progress)

        for c in csvs.values():
            c.close()
