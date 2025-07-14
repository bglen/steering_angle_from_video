import cv2
import numpy as np
from pupil_apriltags import Detector
import csv

# ---- Settings ----
VIDEO_PATH = 'video/test_video_no_tag.mp4'   # Your DJI video file
TAG_FAMILY = 'tag36h11'
TAG_SIZE = 0.05  # meters (set to whatever your printed tag size is)

# Camera intrinsics (estimate for DJI Action 3, or calibrate for more accuracy)
fx = fy = 1000  # approximate focal length in pixels
cx, cy = 960, 540  # image center for 1920x1080

# ---- Initialization ----
print("Initializing...")
cap = cv2.VideoCapture(VIDEO_PATH)
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
with open('steering_angle_output.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Time (s)', 'Steering Angle (deg)'])
    for t, a in output:
        writer.writerow([f"{t:.4f}", "" if a is None else f"{a:.2f}"])

print("Done. Output saved to 'steering_angle_output.csv'")