# Real-Time Vehicle Detection on Raspberry Pi (YOLOv4-tiny)

Traffic-analysis project from my M.Sc. Applied Computer Science
(Hochschule Schmalkalden), graded 1.0 (sehr gut).

Runs on a Raspberry Pi: it reads a traffic video or camera feed,
detects and classifies vehicles (car, bus, truck, motorbike, …) in each
frame with YOLOv4-tiny, and streams the annotated video live to any
browser on the network through a small Flask server.

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
