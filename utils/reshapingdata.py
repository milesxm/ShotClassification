import os 
import numpy as np

#Cover drive paths
cover_drive_path = "Data\poselandmarks\coverdrives"
cover_drive_flipped_path = "Data\poselandmarks\coverdrivesflipped"
cover_drive_augmented_path = "Data\poselandmarks\coverdrivesaugmented"
cover_drive_flipped_augmented_path = "Data\poselandmarks\coverdrivesflippedaugmented"

#Pull shot paths
pull_shot_path = "Data\poselandmarks\pullshots"
pull_shot_flipped_path = "Data\poselandmarks\pullshotsflipped"
pull_shot_augmented_path = "Data\poselandmarks\pullshotsaugmented"
pull_shot_flipped_augmented_path = "Data\poselandmarks\pullshotsflippedaugmented"

#Cut shot paths
cut_shot_path = "Data\poselandmarks\cutshots"
cut_shot_augmented_path = "Data\poselandmarks\cutshotsaugmented"
cut_shot_flipped_path = "Data\poselandmarks\cutshotsflipped"
cut_shot_flipped_augmented_path = "Data\poselandmarks\cutshotsflippedaugmented"

#Sweep shot paths
sweep_shot_path = "Data\poselandmarks\sweepshots"
sweep_shot_flipped_path = "Data\poselandmarks\sweepshotsflipped"
sweep_shot_augmented_path = "Data\poselandmarks\sweepshotsaugmented"
sweep_shot_flipped_augmented_path = "Data\poselandmarks\sweepshotsflippedaugmented"

# Get all the keypoints in a list, to add more just add path to the list
cover_drive_keypoints = [os.path.join(cover_drive_path, file) for file in os.listdir(cover_drive_path)]
cover_drive_flipped_keypoints = [os.path.join(cover_drive_flipped_path, file) for file in os.listdir(cover_drive_flipped_path)]
cover_drive_augmented_keypoints = [os.path.join(cover_drive_augmented_path, file) for file in os.listdir(cover_drive_augmented_path)]
cover_drive_flipped_augmented_keypoints = [os.path.join(cover_drive_flipped_augmented_path, file) for file in os.listdir(cover_drive_flipped_augmented_path)]

#Pull shot keypoints
pull_shot_keypoints = [os.path.join(pull_shot_path, file) for file in os.listdir(pull_shot_path)]
pull_shot_flipped_keypoints = [os.path.join(pull_shot_flipped_path, file) for file in os.listdir(pull_shot_flipped_path)]
pull_shot_augmented_keypoints = [os.path.join(pull_shot_augmented_path, file) for file in os.listdir(pull_shot_augmented_path)]
pull_shot_flipped_augmented_keypoints = [os.path.join(pull_shot_flipped_augmented_path, file) for file in os.listdir(pull_shot_flipped_augmented_path)]

#Cut shot keypoints
cut_shot_keypoints = [os.path.join(cut_shot_path, file) for file in os.listdir(cut_shot_path)]
cut_shot_augmented_keypoints = [os.path.join(cut_shot_augmented_path, file) for file in os.listdir(cut_shot_augmented_path)]
cut_shot_flipped_keypoints = [os.path.join(cut_shot_flipped_path, file) for file in os.listdir(cut_shot_flipped_path)]
cut_shot_flipped_augmented_keypoints = [os.path.join(cut_shot_flipped_augmented_path, file) for file in os.listdir(cut_shot_flipped_augmented_path)]

#Sweep shot keypoints
sweep_shot_keypoints = [os.path.join(sweep_shot_path, file) for file in os.listdir(sweep_shot_path)]
sweep_shot_flipped_keypoints = [os.path.join(sweep_shot_flipped_path, file) for file in os.listdir(sweep_shot_flipped_path)]
sweep_shot_augmented_keypoints = [os.path.join(sweep_shot_augmented_path, file) for file in os.listdir(sweep_shot_augmented_path)]
sweep_shot_flipped_augmented_keypoints = [os.path.join(sweep_shot_flipped_augmented_path, file) for file in os.listdir(sweep_shot_flipped_augmented_path)]

# Combine all the keypoints
all_key_points = cover_drive_keypoints + cover_drive_flipped_keypoints + cover_drive_augmented_keypoints + cover_drive_flipped_augmented_keypoints + pull_shot_keypoints + pull_shot_flipped_keypoints + pull_shot_augmented_keypoints + pull_shot_flipped_augmented_keypoints + cut_shot_keypoints + cut_shot_augmented_keypoints + cut_shot_flipped_keypoints + cut_shot_flipped_augmented_keypoints + sweep_shot_keypoints + sweep_shot_flipped_keypoints + sweep_shot_augmented_keypoints + sweep_shot_flipped_augmented_keypoints

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