import cv2 as cv
import os
import mediapipe as mp
import numpy as np

from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

import matplotlib.pyplot as plt

# MediaPipes drawing function
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

# Key body landmarks for the pose detection model
key_body_landmark_no = [0,11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]
landmark_dict = {
    0: "nose",
    1: "left eye (inner)",
    2: "left eye",
    3: "left eye (outer)",
    4: "right eye (inner)",
    5: "right eye",
    6: "right eye (outer)",
    7: "left ear",
    8: "right ear",
    9: "mouth (left)",
    10: "mouth (right)",
    11: "left shoulder",
    12: "right shoulder",
    13: "left elbow",
    14: "right elbow",
    15: "left wrist",
    16: "right wrist",
    17: "left pinky",
    18: "right pinky",
    19: "left index",
    20: "right index",
    21: "left thumb",
    22: "right thumb",
    23: "left hip",
    24: "right hip",
    25: "left knee",
    26: "right knee",
    27: "left ankle",
    28: "right ankle",
    29: "left heel",
    30: "right heel",
    31: "left foot index",
    32: "right foot index"
}

video_set_path = "Data\Videos\CoverDrives"
output_data_path = "Data\poselandmarks\coverdrives"
model_path = "models\pose_landmarker_lite.task"

vid_index = 0

for video in os.listdir(video_set_path):
    video = os.path.join(video_set_path, video)
    
    video = cv.VideoCapture(video)

    # Get the frames per second (fps)
    fps = video.get(cv.CAP_PROP_FPS)

    # Release the video file

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    # Creating the pose landmarker instance 

    options = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO)
      
    output_images_folder = "Data\poselandmarks\coverdrives"
    outputpose_folder = "outputimages"

    video_keypoints = []


    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_num = 0
        timestamp_ms = 0
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            shape = mp_image.numpy_view().shape

            result = landmarker.detect_for_video(mp_image, timestamp_ms= int(timestamp_ms * (1000 / fps)))

          
            if result.pose_landmarks:
              frame_keypoints = []
              for i in key_body_landmark_no:
                landmark = result.pose_landmarks[0][i]
                frame_keypoints.append([landmark.x, landmark.y, landmark.z])

              video_keypoints.append(frame_keypoints)

          


            timestamp_ms += 1/fps
            frame_num += 1
    
    video_keypoints = np.array(video_keypoints)

    output_file_path = os.path.join(output_data_path, f"{vid_index}.npy")
    np.save(output_file_path, video_keypoints)

    print(f"Saved keypoints for {vid_index} to {output_file_path}")

    vid_index += 1
    video.release()
    cv.destroyAllWindows()