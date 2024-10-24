import cv2
import os

cover_drive_path = r"Data\Videos\CoverDrives"
pull_shot_path = r"Data\Videos\PullShots"
cut_shot_path = r"Data\Videos\CutShots"
sweep_shot_path = r"Data\Videos\SweepShots"

cover_drive_output = r"Data\Videos\CoverDrivesFlipped"
pull_shot_output = r"Data\Videos\PullShotsFlipped"
cut_shot_output = r"Data\Videos\CutShotsFlipped"
sweep_shot_output = r"Data\Videos\SweepShotsFlipped"

cover_drive_vids = [os.path.join(cover_drive_path, file) for file in os.listdir(cover_drive_path)]
pull_shot_vids = [os.path.join(pull_shot_path, file) for file in os.listdir(pull_shot_path)]
cut_shot_vids = [os.path.join(cut_shot_path, file) for file in os.listdir(cut_shot_path)]
sweep_shot_vids = [os.path.join(sweep_shot_path, file) for file in os.listdir(sweep_shot_path)]

def flip_videos(video, output_folder):
    for vid in video:
        print(f"Flipping {vid}")
        video = cv2.VideoCapture(vid)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(output_folder, os.path.basename(vid)), fourcc, fps, (frame_width, frame_height))

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            flipped_frame = cv2.flip(frame, 1)

            out.write(flipped_frame)

        video.release()
        out.release()
        cv2.destroyAllWindows()
    


#flip_videos(cover_drive_vids, cover_drive_output)
#flip_videos(pull_shot_vids, pull_shot_output)
#flip_videos(cut_shot_vids, cut_shot_output)
flip_videos(sweep_shot_vids, sweep_shot_output)