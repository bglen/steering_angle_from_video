import sys
import cv2
import numpy as np
import csv
import cv2.aruco as aruco
from pupil_apriltags import Detector
from pathlib import Path

# ---- Settings ----
TAG_FAMILY = 'tag36h11'
TAG_SIZE = 0.044  # meters (set to whatever your printed tag size is)

# Camera intrinsics (estimate for DJI Action 3, or calibrate for more accuracy)
fx = fy = 1000  # approximate focal length in pixels
cx, cy = 960, 540  # image center for 1920x1080

# Screen Anotation
text_border = 50 # border in pixels from edge of frame

# ---- Select File ----
if len(sys.argv) < 2:
    print("Usage: python safv.py /path/to/video.mp4")
    exit(1)

video_path = sys.argv[1]
video_path_obj = Path(video_path)
csv_path = Path(video_path).with_suffix('.csv')
video_out_path = video_path_obj.with_name(video_path_obj.stem + '_annotated.mp4')

# ---- Initialization ----
print("Initializing...")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f" Failed to open video file: '{video_path}'")
    exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1.0 / fps
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
detector = Detector(families=TAG_FAMILY)
output = []
initial_yaw = None
frame_idx = 0

# --- Setup Video Writer ----
fourcc = cv2.VideoWriter_fourcc(*'avc1')  # For .mp4 output
video_out = cv2.VideoWriter(str(video_out_path), fourcc, fps, (frame_width, frame_height))
if not video_out.isOpened():
    print(f"Failed to open video writer: '{video_out_path}'")
    cap.release()
    exit(1)

# ---- Video Processing ----
print("Processing frames. This may take a while...")
while True:
    # grab the next frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find detections and estimate poses
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=[fx, fy, cx, cy],
        tag_size=TAG_SIZE
    )

    timestamp = frame_idx * dt
    angle_deg = None

    # Draw localization onto video frames
    for det in detections:
        # Draw tag outline (corners are in order: [top-left, top-right, bottom-right, bottom-left])
        corners = det.corners.astype(int)
        for i in range(4):
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[(i + 1) % 4])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw tag ID
        center = tuple(det.center.astype(int))
        cv2.putText(frame, f"ID {det.tag_id}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw pose axis
        if det.pose_R is not None and det.pose_t is not None:
            axis_len = 0.05  # meters
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            dist_coeffs = np.zeros(5)
            rvec, _ = cv2.Rodrigues(det.pose_R)
            tvec = det.pose_t
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, axis_len)

    # Compute steering angle
    if detections:
        # Use the first tag detected (there should only be one)
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

    # Annotate steering angle in lower-left corner
    if angle_deg is not None:
        text = f"Angle: {angle_deg:.2f} deg"
    else:
        text = "Angle: -- deg"
    cv2.putText(
        frame,
        text,
        (text_border, frame_height - text_border),  #  num px from left, num px from top
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),  # white text
        2,
        cv2.LINE_AA
    )

    video_out.write(frame)
    frame_idx += 1

print("Frames processed.")
cap.release()
video_out.release()

# ---- Save to CSV ----
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Time (s)', 'Steering Angle (deg)'])
    for t, a in output:
        writer.writerow([f"{t:.4f}", "" if a is None else f"{a:.2f}"])

print(f"Done. Output saved to: '{csv_path}'")
print(f"Annotated video saved to: '{video_out_path}'")