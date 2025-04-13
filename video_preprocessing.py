import cv2
import os
from models.deepfake_model import DeepfakeModel
import torch

FRAME_FOLDER = 'uploads/frames/'
MODEL_PATH = 'models/deepfake_detector.pt'

# Extract Frames from Video
def extract_frames(video_path, output_folder=FRAME_FOLDER, frame_interval=10):
    """Extract frames from the video at specified intervals."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_index = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            frame_name = f"{output_folder}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_name, frame)
            frame_count += 1

        frame_index += 1

    cap.release()

# Extract label from filename
def extract_label(filename):
    """Extracts 'real' or 'fake' label from the filename."""
    if "real" in filename.lower():
        return "Real"
    elif "fake" in filename.lower():
        return "Fake"
    else:
        return "Unknown"

# Deepfake Detection with Label Comparison
def detect_deepfake_with_labels():
    """Detect deepfakes and compare with actual labels."""
    model = DeepfakeModel(MODEL_PATH)

    results = []
    
    for video_file in os.listdir('uploads/dataset/'):
        actual_label = extract_label(video_file)
        video_path = os.path.join('uploads/dataset/', video_file)

        # Extract frames
        extract_frames(video_path)

        # Perform deepfake detection
        predictions = []
        for frame_file in os.listdir(FRAME_FOLDER):
            frame_path = os.path.join(FRAME_FOLDER, frame_file)
            
            # Model prediction
            prediction = model.predict(frame_path)
            predicted_label = "Fake" if prediction >= 0.5 else "Real"
            
            predictions.append(predicted_label)

        # Majority voting on frames
        real_count = predictions.count("Real")
        fake_count = predictions.count("Fake")
        predicted_video_label = "Real" if real_count > fake_count else "Fake"

        # Store results
        results.append({
            "video": video_file,
            "actual": actual_label,
            "predicted": predicted_video_label,
            "accuracy": "✅" if actual_label == predicted_video_label else "❌"
        })

    return results

