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

"""Save GPS coordinates as KML."""

__author__ = "Levin Gerdes"


import argparse
import os
from typing import Any, List, Tuple

import simplekml  # type: ignore
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr


def get_args() -> argparse.Namespace:
    """Parses CLI arguments"""

    parser = argparse.ArgumentParser(description="")

    # fmt: off
    parser.add_argument("--input-path",  "-i", type=str, default="~/data/bardenas2023/dataset", dest="input_path",  help="Path of input mcaps")
    parser.add_argument("--output-path", "-o", type=str, default="~/data/bardenas2023/extracted", dest="output_path", help="Output will be written here")
    # fmt: on

    return parser.parse_args()


def get_message_and_timestamp(connection, rawdata) -> Tuple[Any, int]:
    """Returns the deserialized message and its timestamp"""
    msg = deserialize_cdr(rawdata, connection.msgtype)
    timestamp = int(str(msg.header.stamp.sec) + str(msg.header.stamp.nanosec).zfill(9))
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

    for index, bag in enumerate(SUBDIRS):
        kml = simplekml.Kml()
        linestring = kml.newlinestring(name=bag)
        coords = []

        with Reader(os.path.join(INPUT_PATH, bag)) as reader:
            print(bag)

            for connection, timestamp, rawdata in reader.messages():
                if connection.topic == "/gnss":
                    msg, time = get_message_and_timestamp(connection, rawdata)
                    coords.append((msg.longitude, msg.latitude))
                    if msg.status.status < 0:
                        print("Message without GNSS Fix")

        linestring.coords = coords
        print("Saving to disk.")
        kml.save(os.path.join(OUTPUT_PATH, bag + ".kml"))
