import os
from preprocessing.video_preprocessing import extract_frames, detect_deepfake

# Test on sample video
video_path = "test_video.mp4"
frames_folder = "uploads/frames"

# Ensure folder exists
os.makedirs(frames_folder, exist_ok=True)

# Extract frames and detect deepfake
extract_frames(video_path, frames_folder)
deepfake_result = detect_deepfake(frames_folder)

print("Deepfake Detection Result:", deepfake_result)
