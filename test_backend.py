#!/usr/bin/env python3
"""
Test the OpenCV YOLO backend directly
"""
import os
import numpy as np
import cv2
from posture_guard import OpenCVYoloBackend

def test_backend():
    # Load the labels
    labels = ['neck_good', 'neck_bad', 'backbone_good', 'backbone_bad', 'buttocks_good', 'buttocks_bad']

    print("Testing OpenCV YOLO Backend...")

    try:
        # Create backend
        backend = OpenCVYoloBackend(labels, "yolov3-obj.cfg", "posture_yolov3.weights")
        print("✓ Backend created successfully")

        # Test with a dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        label, conf, extras = backend.inference(dummy_frame)
        boxes = extras.get('boxes', [])

        print(f"✓ Inference successful:")
        print(f"  Label: {label}")
        print(f"  Confidence: {conf:.3f}")
        print(f"  Boxes: {len(boxes)}")

        # Test with a random colored frame (might trigger some detections)
        random_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        label2, conf2, extras2 = backend.inference(random_frame)
        boxes2 = extras2.get('boxes', [])

        print(f"✓ Random frame inference:")
        print(f"  Label: {label2}")
        print(f"  Confidence: {conf2:.3f}")
        print(f"  Boxes: {len(boxes2)}")

        if boxes2:
            print("  Detected objects:")
            for (x, y, w, h), name, conf in boxes2:
                print(f"    {name}: {conf:.3f} at ({x}, {y}, {w}, {h})")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_backend()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")