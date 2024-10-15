import numpy as np
import os

# Path where your .npy files are stored
output_data_path = "Data\poselandmarks\cutshots"

# List all .npy files
npy_files = [f for f in os.listdir(output_data_path) if f.endswith('.npy')]

# Load and check the shape of each .npy file
for npy_file in npy_files:
    file_path = os.path.join(output_data_path, npy_file)
    data = np.load(file_path)
    print(f"File: {npy_file}, Shape: {data.shape}")