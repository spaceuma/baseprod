# BASEPROD Scripts

These scripts were used to prepare BASEPROD, the Bardenas Semi-Desert Planetary
Rover Dataset.
The code is released under the [MIT License](LICENSE).

Authors:
- Levin Gerdes [![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0001-7648-8928)
- Hugo Leblond [![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0009-0009-2745-0988)
- Rául Castilla Arquillo [![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0003-4203-8069)
- Joaquín Ortega Cortés
- Felix Wilting

Supervisor: Carlos J. Pérez del Pulgar [![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0001-5819-8310)


## Requirements

The code was tested with Python 3.10.12.
Additional Python requirements are listed in `requirements.txt`
and can be installed in a virtual environment as follows:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Activate the environment with `source .venv/bin/activate` or deactivate it by invoking `deactivate`.

Some scripts use additional packages for plotting:
```bash
sudo apt install \
texlive \
texlive-latex-extra \
cm-super \
dvipng
```

The scripts are formatted with [black](https://pypi.org/project/black/) ([exact version](.github/workflows/black.yml#L13)).


## Usage

Download and unzip the (relevant parts of the) dataset to your local drive.
For convenience, we recommend to create a shorthand via `export BASEPROD=/path/to/baseprod`.
This step is assumed for the following example commands.

>[!Note]
>View detailed script usage information via `python <script> --help`.


### Exporting mcap recordings

You can export all data recordings via
```zsh
python export_logs.py -i ${BASEPROD}/rosbags/ -o ${BASEPROD}/rover_sensors/
```
This will iterate over all roslogs in that directory and take ages,
because we write to disk quite often to avoid running out of memory.


### Plotting weather station

To plot weather station data, provide the CSV of the day and data you are interested in.
E.g., to plot all data from the 23rd of July:
```zsh
python plot_weather.py --file ${BASEPROD}/weather_station/2023-07-23_09-49-44.csv --all
```


### Plotting LIBS spectra

Spectra can be plotted by providing a `spectra.csv` and indicating the spectrum one is interested in.
E.g., to plot spectrum 3 of the 2nd measurement at location 1:
```zsh
python plot_spectrum.py --file ${BASEPROD}/libs_measurements/01/02/spectra.csv --spectrum 3
```


### Data corrections

The following commands compute and apply corrections to different measurements.
All input data will be preserved, corrected data can be found in files that carry "CORRECTED" in their filename.

The Force/Torque corrections need an exported calibration log (passed via `-c` flag) and an exported input log to correct (`-i`).
```zsh
python ft_correction.py -c ${BASEPROD}/rover_sensors/calibration/ft_neutral -i ${BASEPROD}/rover_sensors/2023-07-23_12-52-39
```

The bogie offset script only needs an exported input log (`-i`).
```zsh
python bogie_offset.py -i ${BASEPROD}/rover_sensors/2023-07-23_12-52-39
```

The gyro offset script needs an exported input log (`-i`) and can be given a manual offset correction (`-c`) for the heading in radians.
```zsh
python gyro_offset.py -i ${BASEPROD}/rover_sensors/2023-07-23_12-52-39 -c -1.25
```

>[!TIP]
>We processed most data in batch via the following commands.
>```zsh
>export TPATH=/path/to/baseprod/rover_sensors
>ls ${TPATH} | grep 2023 | xargs -I {} python ft_correction.py -c ${TPATH}/calibration/ft_neutral -q -i ${TPATH}/{}
>ls ${TPATH} | grep 2023 | xargs -I {} python bogie_offset.py -q -i ${TPATH}/{}
>ls ${TPATH} | grep 2023 | xargs -I {} python gyro_offset.py -q -i ${TPATH}/{}
>```
