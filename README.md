Advanced Computer Vision Techniques with OpenCV

This repository contains implementations of advanced computer vision techniques using OpenCV 3 and MediaPipe. The code was developed step by step while following tutorials, providing a hands-on approach to understanding key concepts and functionalities in computer vision.
Features

🧑‍💻 Face Mesh Detection

    Utilizes MediaPipe Face Mesh for precise facial landmark detection.
    Detects 468 facial landmarks in real time.
    Converts images from BGR to RGB format for proper processing.
    Overlays facial landmarks and tracks them dynamically.
    Provides landmark coordinates for further analysis or applications.

🏃 Pose Detection

    Implements MediaPipe Pose Estimation for real-time human pose tracking.
    Detects 33 body keypoints (including head, torso, and limbs).
    Draws skeleton connections using MediaPipe's POSE_CONNECTIONS.
    Converts BGR images to RGB for accurate landmark detection.
    Visualizes keypoints with distinct markers for different body parts.

🖐 Hand Detection & Tracking

    Uses MediaPipe Hands for real-time hand tracking.
    Detects and tracks multiple hands with 21 key landmarks per hand.
    Highlights specific landmarks (e.g., fingertips) for gesture recognition.
    Draws bounding boxes and landmarks with customizable line thickness and colors.

😊 Face Detection

    Implements MediaPipe Face Mesh for real-time facial landmark detection.
    Detects multiple faces and accurately tracks their movements.
    Draws bounding boxes and key facial features for visualization.
    Provides real-time face tracking and recognition.



Technologies Used

    OpenCV – Real-time computer vision
    MediaPipe – Face, Hand, and Pose detection
    Python (with libraries such as cv2, time)
