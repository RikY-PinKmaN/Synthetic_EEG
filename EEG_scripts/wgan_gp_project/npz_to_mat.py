import numpy as np
from scipy.io import savemat
import argparse

def convert_npz_to_mat(npz_file, mat_file):
    """
    Converts a .npz file to a .mat file.

    Args:
        npz_file (str): The path to the input .npz file.
        mat_file (str): The path to the output .mat file.
    """
    try:
        # Load the data from the .npz file
        data = np.load(npz_file)
        
        # Create a dictionary to store the data for the .mat file
        mat_dict = {key: data[key] for key in data}
        
        # Save the data to a .mat file
        savemat(mat_file, mat_dict)
        
        print(f"Successfully converted '{npz_file}' to '{mat_file}'")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Convert a .npz file to a .mat file.")
    parser.add_argument("npz_file", help="The path to the input .npz file.")
    parser.add_argument("mat_file", help="The path to the output .mat file.")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Convert the file
    convert_npz_to_mat(args.npz_file, args.mat_file)
