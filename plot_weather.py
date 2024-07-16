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
Plot weather station information.

Call the function with:
python plot_weather --file <csv> --[desired data]

Get an overview of the options with:
    python plot_weather --help
"""

__author__ = "Raúl Castilla Arquillo"


import argparse
import csv
import os.path
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()

parser.add_argument(
    "--file", type=str, required=True, help="the file of weather information to plot"
)

parser.add_argument(
    "--radiation",
    action="store_true",
    help="shows solar radition plots",
)

parser.add_argument(
    "--temperature",
    action="store_true",
    help="shows air temperature plots",
)

parser.add_argument(
    "--humidity",
    action="store_true",
    help="shows air humidity plots",
)

parser.add_argument(
    "--pressure",
    action="store_true",
    help="shows air pressure plots",
)

parser.add_argument(
    "--all",
    action="store_true",
    help="shows plots for all the weather station information: sun radiation, temperature, humidity and pressure",
)

parser.add_argument(
    "--latex",
    "-l",
    action="store_true",
    dest="latex",
    help="Render text in Latex",
)


def plot_weather_data(ax, date_str, data, x_label, y_label, title):
    ax.plot(date_str, data)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    # Date format for x axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    # Rotating xticks labels
    ax.figure.autofmt_xdate()
    ax.grid()


if __name__ == "__main__":
    args = parser.parse_args()
    file_basename = os.path.basename(args.file)

    plt.rcParams["svg.fonttype"] = "none"
    if args.latex:
        plt.rcParams.update(
            {"text.usetex": True, "font.family": "Computer Modern Roman"}
        )

    time_str = []
    solar_radiation = []
    air_temperature = []
    air_humidity = []
    air_pressure = []

    with open(args.file, "r") as csvfile:
        plots = csv.reader(csvfile, delimiter=",")
        headers = next(csvfile)

        for row in plots:
            time_str.append(row[0])
            solar_radiation.append(float(row[1]))
            air_temperature.append(float(row[2]))
            air_humidity.append(float(row[3]))
            air_pressure.append(float(row[4]))

        date = file_basename[:10]
        datetime_str = [datetime.strptime(time, "%H:%M:%S") for time in time_str]

        if args.radiation:
            fig, axs = plt.subplots(1, 1, figsize=(8, 8))
            plot_weather_data(
                axs,
                datetime_str,
                solar_radiation,
                "Time (HH:MM:SS)",
                r"Irradiance $\left[\frac{W}{m^2}\right]$",
                date + "\nSolar radiation",
            )
        elif args.temperature:
            fig, axs = plt.subplots(1, 1, figsize=(8, 8))
            plot_weather_data(
                axs,
                datetime_str,
                air_temperature,
                "Time (HH:MM:SS)",
                r"Temperature $[\degree C]$",
                date + "\nAir Temperature",
            )
        elif args.humidity:
            fig, axs = plt.subplots(1, 1, figsize=(8, 8))
            plot_weather_data(
                axs,
                datetime_str,
                air_humidity,
                "Time (HH:MM:SS)",
                "Humidity $[\%]$",
                date + "\nAir Humidity",
            )
        elif args.pressure:
            fig, axs = plt.subplots(1, 1, figsize=(8, 8))
            plot_weather_data(
                axs,
                datetime_str,
                air_pressure,
                "Time (HH:MM:SS)",
                "Pressure [hPa]",
                date + "\nAir Pressure",
            )
        elif args.all:
            fig, axs = plt.subplots(4, 1, figsize=(8, 12))
            fig.suptitle("Weather information, " + date, fontsize=16)
            plot_weather_data(
                axs[0],
                datetime_str,
                solar_radiation,
                "Time (HH:MM:SS)",
                r"Irradiance $\left[\frac{W}{m^2}\right]$",
                "Solar radiation",
            )
            plot_weather_data(
                axs[1],
                datetime_str,
                air_temperature,
                "Time (HH:MM:SS)",
                r"Temperature $[\degree C]$",
                "Air Temperature",
            )
            plot_weather_data(
                axs[2],
                datetime_str,
                air_humidity,
                "Time (HH:MM:SS)",
                "Humidity $[\%]$",
                "Air Humidity",
            )
            plot_weather_data(
                axs[3],
                datetime_str,
                air_pressure,
                "Time (HH:MM:SS)",
                "Pressure [hPa]",
                "Air Pressure",
            )
        else:
            print("Please, define what type of data you wish to plot.")
            parser.print_help()
            parser.exit()

        plt.show()
