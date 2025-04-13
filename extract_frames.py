import cv2
import os

VIDEO_DIR = "uploads/dataset"
OUTPUT_DIR = "uploads/frames"
FRAME_INTERVAL = 30  # extract one frame every 30 frames

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_frames():
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    print(f"[INFO] Found {len(video_files)} video(s) in {VIDEO_DIR}")

    for video in video_files:
        label = "fake" if "fake" in video.lower() else "real"
        video_path = os.path.join(VIDEO_DIR, video)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        saved_count = 0
        success = True

        while success:
            success, frame = cap.read()
            if not success:
                break

            if frame_count % FRAME_INTERVAL == 0:
                frame_filename = f"{label}_{os.path.splitext(video)[0]}_frame_{frame_count}.jpg"
                frame_path = os.path.join(OUTPUT_DIR, frame_filename)
                cv2.imwrite(frame_path, frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"[INFO] Extracted {saved_count} frame(s) from {video}")

if __name__ == "__main__":
    extract_frames()
