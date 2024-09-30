import os 
import numpy as np

cover_drive_path = "Data\poselandmarks\coverdrives"
pull_shot_path = "Data\poselandmarks\pullshots"


# Get all the keypoints in a list, to add more just add path to the list
cover_drive_keypoints = [os.path.join(cover_drive_path, file) for file in os.listdir(cover_drive_path)]
pull_shot_keypoints = [os.path.join(pull_shot_path, file) for file in os.listdir(pull_shot_path)]

all_key_points = cover_drive_keypoints + pull_shot_keypoints

max_frames = 0

for keypoints in all_key_points:
    keypoints = np.load(keypoints)
    sequence_length = keypoints.shape[0]

    if sequence_length > max_frames:
        max_frames = sequence_length

# Padding the videos

def pad(data,max_frames):
    seq_length = data.shape[0]

    if seq_length < max_frames:
        padding = np.zeros((max_frames - seq_length, data.shape[1], data.shape[2]))
        data = np.vstack([data, padding])

    return data


for keypoints in all_key_points:
    data = np.load(keypoints)
    padded_keypoints = pad(data, max_frames)
    np.save(keypoints, padded_keypoints)

    print(f"Padded and saved {keypoints}")