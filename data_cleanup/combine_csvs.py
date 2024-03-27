"""
Combines multiple CSV files into a single, large one. Assumes sibling directory
'data' with subdirectory "raw_data" containing CSVs to be combined.

Authors: 
- Rani Hinnawi

Sources:
- GeeksForGeeks
- Bing Copilot

Last updated: 2024-03-22
"""

from typing import List
import os

def merge_csv_files(file_list: List[str], output_csv_path: str):
    """
    Merges CSV files from a provided list into a single, large file. Outputs
    new CSV file into user-provided path.

    Args:
        file_list (list): List of CSV file paths
        output_csv_path (str): Combined CSV file path
    """
    with open(output_csv_path, 'w') as output_file:
        # Write header once
        header_written = False

        # Iterate through file list and append each file to new, output file
        for file in file_list:
            with open(file, 'r') as input_file:
                # Case: header already written
                if header_written:
                    next(input_file)
                else:
                    header_written = True

                # Append each line from original CSV files into new one
                for line in input_file:
                    output_file.write(line)

    print(f"Merger complete. Data saved to {output_csv_path}")

if __name__ == "__main__":
    csv_files = ['../data/raw_data_cleaned/' + file for file in \
                 os.listdir('../data/raw_data_cleaned') if \
                    file.endswith('.csv')]
    output_csv = '../data/cleaned_combined_data.csv'
    merge_csv_files(csv_files, output_csv)