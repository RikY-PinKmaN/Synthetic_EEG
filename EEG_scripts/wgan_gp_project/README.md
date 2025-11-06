# WGAN-GP EEG Synthetic Data Generator

This project uses WGAN-GP to generate synthetic EEG data for motor imagery tasks.

## Installation
1. Clone the repository.
2. Install dependencies with Poetry:

## Usage
1. Place your `.mat` file in the `data/` directory.
2. Run the main script:

## Project Structure
- `src/`: Contains the code for preprocessing, modeling, training, and evaluation.
- `data/`: Store input data files.
- `main.py`: Orchestrates the data loading, preprocessing, and training.

## Dependencies
- TensorFlow
- NumPy
- SciPy
- Scikit-learn
- MNE
- Matplotlib
