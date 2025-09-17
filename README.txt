Posture Guard — Side View (Local Desktop App)

Quick install
-------------
1) Create/activate a Python 3.10+ environment (recommended).
2) Install dependencies:

   pip install -U pip
   pip install PyQt6 opencv-python onnxruntime numpy pyttsx3 mediapipe
   # Only if you will use .pt YOLO models:
   # pip install ultralytics

Files to have ready
-------------------
- posture_guard.py        (the app)
- posture.names           (your 6 labels; already included here)
- YOUR_MODEL.cfg          (your YOLOv3 cfg — the one you pasted)
- posture_yolov3.weights  (your trained weights)

Run the app
-----------
python posture_guard.py

Inside the app
--------------
Model → Load labels (.names)          -> pick posture.names
Model → Load YOLO cfg + weights…      -> pick YOUR_MODEL.cfg then posture_yolov3.weights
Camera → Start                        -> sit side-on; slouch to test the alert

Tips
----
- The app treats any class that ends with "_bad" as a BAD posture signal (triggers warning/voice).
- Your cfg uses width=256 height=256 — OpenCV backend will run at that size automatically.
- If nothing is detected, double-check that posture.names has exactly 6 lines in the same order used at training.

Build a one-file executable (optional)
--------------------------------------
pip install pyinstaller
pyinstaller --noconfirm --name PostureGuard --onefile --windowed posture_guard.py

The generated binary will run fully offline.
