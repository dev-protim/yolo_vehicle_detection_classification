# Real-Time Vehicle Detection with YOLOv4-tiny

Part of my M.Sc. Applied Computer Science traffic-analysis project
(Hochschule Schmalkalden) [– final project graded 1.0, ran on a Raspberry Pi].

The app reads a traffic video (or a camera), detects vehicles in every
frame with YOLOv4-tiny, and streams the annotated video live to the
browser through a small Flask server.

## How it works
1. OpenCV's DNN module loads the YOLOv4-tiny model (COCO classes:
   car, bus, truck, motorbike, bicycle, …).
2. Each frame is resized to 416×416 and passed through the network.
3. Detections above 50 % confidence are kept; Non-Maximum Suppression
   removes duplicate boxes.
4. Boxes, class labels and confidence scores are drawn on the frame.
5. Frames are encoded as JPEG and streamed as MJPEG at `/video`.

## Tech stack
Python · OpenCV (DNN) · YOLOv4-tiny · NumPy · Flask

## Run locally
1. Install dependencies:
   pip install -r requirements.txt
2. Download `yolov4-tiny.weights` (not included, too large for Git) from
   the official Darknet / AlexeyAB YOLOv4 release and put it in the
   project folder.
3. Start the server:
   python project.py
4. Open http://localhost:5000/video

To use a live camera instead of the sample video, change
`video_path = 'traffic_video.mp4'` to `0` in `project.py`.

## Files
- `project.py` – detection and streaming server
- `yolov4-tiny.cfg`, `coco.names` – model config and class names
- `traffic_video.mp4` – sample traffic video

## Results
https://drive.google.com/drive/folders/1Xv9uZJCAOw0oAeRJT5aCDBtyQ2AdzhTG?hl=DE
