import numpy as np
import os
import torch
import mediapipe as mp
import cv2

import torch.nn as nn
import torch.nn.functional as F

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions

def process_new_vid(video_path, media_pipe_model_path):
    model_path = media_pipe_model_path
    video_path = video_path

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    # Creating the pose landmarker instance 

    options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO)

    

    key_body_landmark_no = [0,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]

    video_keypoints = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_num = 0
        timestamp_ms = 0


        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            mp_image = mp.Image(image_format= mp.ImageFormat.SRGB, data=frame)
            shape = mp_image.numpy_view().shape

            result = landmarker.detect_for_video(mp_image, timestamp_ms = int(timestamp_ms * 1000/fps))
            
            normalised_coords = []

            filtered_landmarks = []



            if result.pose_landmarks:
                frame_keypoints = []
                for i in key_body_landmark_no:
                    landmark = result.pose_landmarks[0][i]
                    frame_keypoints.append([landmark.x, landmark.y, landmark.z])

                video_keypoints.append(frame_keypoints)

            timestamp_ms += 1/fps
            frame_num += 1

    video.release()
    cv2.destroyAllWindows()
                
    video_keypoints = np.array(video_keypoints)

    return video_keypoints




class CricketShotClassifier(nn.Module):
    def __init__(self):
        super(CricketShotClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=51, hidden_size=256, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(256, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 3)  # Two shot types (cover drive, pull shot)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Use the hidden state of the last time step
        x = self.fc1(lstm_out)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load the model
model = CricketShotClassifier()
model.load_state_dict(torch.load('cricketshotclassifierv4.2.pth'))
model.eval()




def pad_vid(keypoints, max_frames = 109):
    num_frames = keypoints.shape[0]

    if num_frames < max_frames:
        pad = max_frames - num_frames
        keypoints = F.pad(torch.tensor(keypoints), (0, 0, 0, 0, 0, pad), "constant", 0)
    else:
        keypoints = keypoints[:max_frames,:,:]

    keypoints = keypoints.view(max_frames, -1)

    return keypoints


video_path = "Lefty Cut.mp4"

#can choose model here
video_keypoints = process_new_vid(video_path, "models\pose_landmarker_heavy.task")

print(video_keypoints.shape)



video_keypoints = pad_vid(video_keypoints)

video_keypoints = torch.tensor(video_keypoints, dtype=torch.float32)

video_keypoints = video_keypoints.unsqueeze(0)

with torch.no_grad():
    # Get the model's output (softmax will already be applied in your model's forward function)
    prediction = model(video_keypoints)

    print(prediction)

    # Get the predicted class (0 or 1 or 2)
    predicted_class = torch.argmax(prediction, dim=1)

    

    # Get the confidence for the predicted class
    confidence = prediction[0][predicted_class].item() * 100  # Multiply by 100 to get percentage

# Print the predicted class with confidence
if predicted_class == 0:
    print(f"Cover Drive with {confidence:.2f}% confidence")
elif predicted_class == 1:
    print(f"Pull Shot with {confidence:.2f}% confidence")
else:
    print(f"Cut Shot with {confidence:.2f}% confidence")
