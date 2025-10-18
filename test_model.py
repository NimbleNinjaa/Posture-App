#!/usr/bin/env python3
"""
Test script to validate YOLO model loading and basic inference
"""
import os
import sys
import cv2
import numpy as np

def test_yolo_model():
    """Test if the YOLO model loads and can perform inference"""

    # Check if files exist
    cfg_path = "yolov3-obj.cfg"
    weights_path = "posture_yolov3.weights"
    names_path = "posture .names"

    print("Checking files:")
    for path in [cfg_path, weights_path, names_path]:
        exists = os.path.exists(path)
        print(f"  {path}: {'✓' if exists else '✗'}")
        if not exists:
            print(f"ERROR: Required file {path} not found!")
            return False

    # Load class names
    labels = []
    try:
        with open(names_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
        print(f"\nLoaded {len(labels)} class labels: {labels}")
    except Exception as e:
        print(f"ERROR loading labels: {e}")
        return False

    # Parse config for network size
    net_w, net_h = 608, 608
    classes_in_cfg = None
    try:
        with open(cfg_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip().split('=')
                if len(s) == 2:
                    k, v = s[0].strip().lower(), s[1].strip()
                    if k == 'width':
                        net_w = int(v)
                    elif k == 'height':
                        net_h = int(v)
                    elif k == 'classes':
                        classes_in_cfg = int(v)
        print(f"Network size from config: {net_w}x{net_h}")
        print(f"Classes in config: {classes_in_cfg}")
    except Exception as e:
        print(f"WARNING: Could not parse config file: {e}")

    # Validate class count
    if classes_in_cfg is not None and classes_in_cfg != len(labels):
        print(f"WARNING: Config specifies {classes_in_cfg} classes but found {len(labels)} labels")

    # Try to load the model
    print("\nLoading YOLO model...")
    try:
        net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        print("✓ Model loaded successfully!")

        # Get output layer names
        output_layers = net.getUnconnectedOutLayersNames()
        print(f"Output layers: {output_layers}")

    except Exception as e:
        print(f"✗ ERROR loading model: {e}")
        return False

    # Test inference with a dummy image
    print("\nTesting inference...")
    try:
        # Create a test image (640x480, typical webcam resolution)
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Prepare blob
        blob = cv2.dnn.blobFromImage(test_img, 1/255.0, (net_w, net_h), swapRB=True, crop=False)
        net.setInput(blob)

        # Run inference
        outs = net.forward(output_layers)
        print(f"✓ Inference successful! Got {len(outs)} output tensors")

        # Analyze outputs
        total_detections = 0
        for i, out in enumerate(outs):
            print(f"  Output {i}: shape {out.shape}")
            # Count detections with confidence > 0.1
            for det in out:
                det = np.array(det)
                if len(det) >= 6:  # Make sure we have class scores
                    scores = det[5:]
                    max_conf = float(np.max(scores))
                    if max_conf > 0.1:
                        total_detections += 1

        print(f"Found {total_detections} detections with confidence > 0.1")

    except Exception as e:
        print(f"✗ ERROR during inference: {e}")
        return False

    print("\n✓ All tests passed! Your model should work with the application.")
    return True

if __name__ == "__main__":
    success = test_yolo_model()
    sys.exit(0 if success else 1)