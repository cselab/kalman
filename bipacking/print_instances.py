import pickle
import numpy as np
import pandas as pd

def print_instances(file_path):
    """Loads and prints the content of a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            instances = pickle.load(f)
            # This is necessary to display the full DataFrame content.
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            print(instances)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    print_instances('bipacking/train_instances.pkl')
