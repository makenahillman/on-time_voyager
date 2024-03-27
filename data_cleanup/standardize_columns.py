"""
Standardize column data types to reduce memory usage and enhance usability for
model and back-end. Assumes sibling directory 'data' with subdirectories 
raw_data and test_data.

Authors:
- Rani Hinnawi

Sources:

Last updated: 2024-03-22
"""

import pandas as pd
import os

def standardize_columns(csv_file: (str), output_csv_path: str):
    """
    Converts data types for all columns into more compressed equivalents
    without altering original values. Remove unwanted columns.

    Args:
        csv_file (list): CSV file path
        output_csv_path (str): Path of output CSV file
    """
    # Set up which columns to drop or convert (by dtype)
    drop_columns = ["WHEELS_ON", "WHEELS_OFF", "Unnamed: 27"]

    # Int64 type preserves empty (NaN) cells. Int8 and Int16 do not
    convert_to_int64 = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", \
                        "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
    convert_to_int16 = [\
        "OP_CARRIER_FL_NUM", "CRS_DEP_TIME", "DEP_TIME", "CRS_ARR_TIME", \
            "ARR_TIME", "DEP_DELAY", "ARR_DELAY", "CRS_ELAPSED_TIME", \
                "ACTUAL_ELAPSED_TIME", "AIR_TIME", "DISTANCE"] 
    convert_to_int8 = ["CANCELLED", "DIVERTED", "TAXI_OUT", "TAXI_IN"]

    # Drop rows with missing times - required data
    drop_na = [\
        "FL_DATE", "OP_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN", "DEST", \
            "CRS_DEP_TIME", "DEP_TIME", "CRS_ARR_TIME", "ARR_TIME", "DEP_DELAY", \
            "ARR_DELAY", "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME",\
                  "DISTANCE",  "TAXI_OUT", "TAXI_IN"]

    # Create df
    df = pd.read_csv(csv_file)

    # Drop designated columns and rows with missing important data values
    df.drop(drop_columns, axis=1, inplace=True)
    df.dropna(subset=drop_na, inplace=True)

    # Convert float columns to int16 and int8
    df[convert_to_int64] = df[convert_to_int64].astype('Int64')
    df[convert_to_int16] = df[convert_to_int16].fillna(0).astype('int16')
    df[convert_to_int8] = df[convert_to_int8].fillna(0).astype('int8')

    df.to_csv(output_csv_path)
    print(f"Column update complete. Data saved to {output_csv_path}")

if __name__ == "__main__":
    # # Uncomment to test
    csv_file = '../data/test_data/2009_test.csv'
    output_csv = '../data/test_data/2009_test2_output.csv'
    standardize_columns(csv_file, output_csv)
    
    # csv_files = [file for file in os.listdir('../data/raw_data') if file.endswith('.csv')]
    # csv_output_files = ["../data/raw_data_cleaned/cleaned_" + file for file in csv_files]
    # csv_files = ["../data/raw_data/" + file for file in csv_files]
    
    # for csv_file, output_csv in zip(csv_files, csv_output_files):
    #     standardize_columns(csv_file, output_csv)