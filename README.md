# On Time Voyager

## Project Details:

&nbsp; CSE 6242 Team Project Code
<br />
&nbsp; Team 025: Taylor Boyd, Jasmine Camacho, Michael Firn, Riley Harkins, Makena Hillman, Rani Hinnawi

## Structure

- data/
  - raw_data/
  - raw_data_cleaned/
  - test_data/
- data_clean_up/
- model/
- visualization/

## Setup

1. Run `git clone https://github.com/makenahillman/on-time_voyager.git`
1. If directory structure does not much above, add empty folders as needed
1. Create and activate conda environment using `conda create --name <my_env> python=3.10`
1. Activate conda environment: `conda activate <my_env>`
1. Install dependencies from sub-project (data clean-up, model, or visualization): `pip install -r <dir_path>/requirements.txt`
1. If working with data CSV files, download from Kaggle (links below) relevant files
   1. Full sets will be named 'cleaned_combined_data<n>.csv', where n is a version number
   1. Raw, original data is linked under the 'Original data' sub-section as 'Airline Delay and Cancellation Data, 2009 - 2018'

If pushing code changes to Github, please create a new branch either locally or on the [Github project page](https://github.com/makenahillman/on-time_voyager). The directions above pull from the 'main' branch.

## Dependencies:

#### Pandas, Matplotlib, Sklearn

&nbsp; All downloadable with pip

## Dataset:

#### Cleaned dataset can be found on Kaggle:

- [Kaggle data sets (CSVs)](https://kaggle.com/datasets/7b34d1b00b68c879e7409db0bcab36f35f0b0ff6427cd44a01494d84c85086fe)
- [Most current, full data set](https://www.kaggle.com/datasets/ranihinnawi/flight-data-2009-2018-combined/data?select=cleaned_combined_data2.csv)

#### Original data

- [Airline Delay and Cancellation Data, 2009 - 2018](https://www.kaggle.com/datasets/yuanyuwendymu/airline-delay-and-cancellation-data-2009-2018/data)

## Sources

- [Kaggle Dataset Collaboration](https://www.kaggle.com/docs/datasets#collaborating-on-datasets)
- [Column definitions for related data set](https://www.stat.purdue.edu/~lfindsen/stat350/airline2008_dataset_definition.pdf)
