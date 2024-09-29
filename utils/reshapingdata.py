import torch
import numpy as np

# Load your npy files
keypoints = np.load('your_keypoints_file.npy')

# Convert to torch tensors
keypoints_tensor = torch.tensor(keypoints)

# Pad the sequence with zeros (or a specific value)
padded_keypoints = torch.nn.functional.pad(keypoints_tensor, (0, 0, 0, max_length - keypoints_tensor.shape[0]))

print(padded_keypoints.shape)  # Should now be consistent across all videos
