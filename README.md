# YOLOv8 People Counter (Entry / Exit Tracking)

A computer vision project that uses YOLOv8 and object tracking to count how many people enter and exit a scene from a video stream. The system tracks individuals across frames, detects when they cross a virtual vertical line, and updates live counters on the video feed.

This project focuses on real-time detection, tracking, and simple motion-based logic, and was built as a practical OpenCV and deep learning exercise.

---

## Features

* Person detection using YOLOv8
* Object tracking with persistent IDs
* Entry and exit detection via line crossing
* Live on-screen counters (Entered / Exited)
* Works on prerecorded video files
* Bounding boxes, centroids, and track IDs drawn on frames

---

## How It Works

1. YOLOv8 detects people (class_id == 0) in each frame
2. The tracker assigns a unique ID to each person
3. The centroid of each bounding box is computed
4. Previous and current centroids are compared
5. If a centroid crosses the vertical reference line:

   * Left to right: Entered
   * Right to left: Exited
6. Counters are updated and displayed on screen

---

## Project Structure

```
project-root/
│
├── main.py              # Main detection and tracking loop
├── drawers.py           # Drawing logic and line-crossing detection
├── videos/
│   └── handlep.mp4      # Input video file
├── yolov8n.pt           # YOLOv8 pretrained weights
```

---

## How to Run

### 1. Install dependencies

```bash
pip install ultralytics opencv-python
```

### 2. Run the program

```bash
python main.py
```

Press ESC to exit the video window.

---

## Configuration

### Change the counting line position

Inside drawers.py:

```python
line_x = 400
```

Adjust this value depending on the video resolution and camera angle.

---

## Key Files Explained

### main.py

* Loads the YOLOv8 model
* Reads video frames
* Calls detection and tracking
* Displays live counters

### drawers.py

* Draws bounding boxes and IDs
* Maintains centroid history per track ID
* Detects line crossing events
* Draws counters and UI elements

---

## Limitations

* Works best with side-view entry points
* Assumes a single vertical entry line
* No re-identification across camera resets
* Not optimized for crowded scenes

---

## Possible Improvements

* Integrate Deep SORT or ByteTrack explicitly
* Support multiple entry and exit lines
* Add region-of-interest (ROI) selection
* Export counts to a file or database
* Support live webcam or RTSP streams

---

## Author

Built as a personal computer vision project using YOLOv8 and OpenCV.

---

## Notes

This project emphasizes clarity and correctness over complexity and is intended as a learning-focused implementation rather than a production system.
