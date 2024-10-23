import numpy as np
import os 
import torch

def guassian_noise(keypoints, noise_std):
    noise = np.random.normal(0, noise_std, keypoints.shape)
    noisy_keypoints = keypoints + noise

    return noisy_keypoints

cover_drive_path = "Data\poselandmarks\coverdrivesflipped"
pull_shot_path = "Data\poselandmarks\pullshotsflipped"
cutshot_path = "Data\poselandmarks\cutshots"
swepshot_path = "Data\poselandmarks\sweepshots"

cover_drive_keypoints = [os.path.join(cover_drive_path, file) for file in os.listdir(cover_drive_path)]
pull_shot_keypoints = [os.path.join(pull_shot_path, file) for file in os.listdir(pull_shot_path)]   
cut_shot_keypoints = [os.path.join(cutshot_path, file) for file in os.listdir(cutshot_path)]
sweep_shot_keypoints = [os.path.join(swepshot_path, file) for file in os.listdir(swepshot_path)]


keypoints_idx = 0
#for keypoints in cover_drive_keypoints:
    #data = np.load(keypoints)
    #noisy_keypoints = guassian_noise(data, 0.05)
    #np.save(f'Data\poselandmarks\coverdrivesaugmented\cda{keypoints_idx}', noisy_keypoints)
    #print(f"Added noise to {noisy_keypoints}")
    #keypoints_idx += 1

keypoints_idx = 0

#for keypoints in pull_shot_keypoints:
    #data = np.load(keypoints)
    #noisy_keypoints = guassian_noise(data, 0.05)
    #np.save(f'Data\poselandmarks\pullshotsaugmented\psa{keypoints_idx}', noisy_keypoints)
    #print(f"Added noise to {noisy_keypoints}")
    #keypoints_idx += 1


#for keypoints in cut_shot_keypoints:
 #   data = np.load(keypoints)
  #  noisy_keypoints = guassian_noise(data, 0.05)
   # np.save(f'Data\poselandmarks\cutshotsaugmented\csa{keypoints_idx}', noisy_keypoints)
    #print(f"Added noise to {noisy_keypoints}")
    #keypoints_idx += 1

for keypoints in pull_shot_keypoints:
    data = np.load(keypoints)
    noisy_keypoints = guassian_noise(data, 0.05)
    np.save(f'Data\poselandmarks\pullshotsflippedaugmented\psfa{keypoints_idx}', noisy_keypoints)
    print(f"Added noise to {noisy_keypoints}")
    keypoints_idx += 1