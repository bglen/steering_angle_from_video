# steering_angle_from_video

![Example of annotated video](/docs/annotated_example.png)

Camera-based steering angle estimation using AprilTags. This script analyzes a video file, extracts the steering wheel angle, saves the data to a timestamped CSV file, and also provides an annotated video.

# Usage

1. Clone repo
2. Create virtual enviornment
3. Install dependencies using the requirements.txt

Run the script as `python safv.py [path/to/video]`

You can set an optional flag `-o [origin timestamp]` to tell the script where in the video should the zero angle for the steering wheel be referenced from. It will use the nearest frame where the AprilTag pose is resolved. Without the origin timestamp, the first frame with a resolved pose is set as the zero point.
