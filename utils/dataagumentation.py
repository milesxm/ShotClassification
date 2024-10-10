import numpy as np
import os 
import torch

def guassian_noise(keypoints, noise_std):
    noise = np.random.normal(0, noise_std, keypoints.shape)
    noisy_keypoints = keypoints + noise

    return noisy_keypoints

cover_drive_path = "Data\poselandmarks\coverdrives"
pull_shot_path = "Data\poselandmarks\pullshots"

cover_drive_keypoints = [os.path.join(cover_drive_path, file) for file in os.listdir(cover_drive_path)]
pull_shot_keypoints = [os.path.join(pull_shot_path, file) for file in os.listdir(pull_shot_path)]   


keypoints_idx = 0
for keypoints in cover_drive_keypoints:
    data = np.load(keypoints)
    noisy_keypoints = guassian_noise(data, 0.05)
    np.save(f'Data\poselandmarks\coverdrivesaugmented\cda{keypoints_idx}', noisy_keypoints)
    print(f"Added noise to {noisy_keypoints}")
    keypoints_idx += 1

keypoints_idx = 0

for keypoints in pull_shot_keypoints:
    data = np.load(keypoints)
    noisy_keypoints = guassian_noise(data, 0.05)
    np.save(f'Data\poselandmarks\pullshotsaugmented\psa{keypoints_idx}', noisy_keypoints)
    print(f"Added noise to {noisy_keypoints}")
    keypoints_idx += 1