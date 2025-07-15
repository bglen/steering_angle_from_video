import sys
import cv2
import numpy as np
import csv
from pupil_apriltags import Detector
from pathlib import Path

# ---- Settings ----
TAG_FAMILY = 'tag36h11'
TAG_SIZE = 0.05  # meters (set to whatever your printed tag size is)

# Camera intrinsics (estimate for DJI Action 3, or calibrate for more accuracy)
fx = fy = 1000  # approximate focal length in pixels
cx, cy = 960, 540  # image center for 1920x1080

# ---- Select File ----
if len(sys.argv) < 2:
    print("Usage: python your_script.py /path/to/video.mp4")
    exit(1)

video_path = sys.argv[1]
csv_path = Path(video_path).with_suffix('.csv')

# ---- Initialization ----
print("Initializing...")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1.0 / fps
detector = Detector(families=TAG_FAMILY)
output = []
initial_yaw = None
frame_idx = 0

# ---- Video Processing ----
print("Processing frames. This may take a while...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=[fx, fy, cx, cy],
        tag_size=TAG_SIZE
    )

    timestamp = frame_idx * dt
    angle_deg = None

    if detections:
        # Use the first tag detected
        det = detections[0]

        # Get rotation matrix (3x3)
        R = det.pose_R

        # Compute yaw angle (rotation around Z axis of tag)
        # Convert to Euler angles (ZYX)
        yaw = np.arctan2(R[1, 0], R[0, 0])  # In radians

        # Store initial reference
        if initial_yaw is None:
            initial_yaw = yaw

        # Relative angle in degrees
        angle_deg = np.degrees(yaw - initial_yaw)

    output.append((timestamp, angle_deg))
    frame_idx += 1

print("Frames processed.")
cap.release()

# ---- Save to CSV ----
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Time (s)', 'Steering Angle (deg)'])
    for t, a in output:
        writer.writerow([f"{t:.4f}", "" if a is None else f"{a:.2f}"])

print(f"Done. Output saved to: '{csv_path}'")