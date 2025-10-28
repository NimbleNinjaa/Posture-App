# Posture Guard

<div align="center">

**Your Personal AI-Powered Posture Assistant**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Open%20Source-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](https://github.com/yourusername/posture-app)

Real-time posture monitoring desktop application that helps you maintain good posture during long work or gaming sessions.

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [How It Works](#how-it-works) • [Contributing](#contributing)

</div>

---

## Table of Contents

- [Overview](#overview)
- [The Problem](#the-problem)
- [The Solution](#the-solution)
- [Features](#features)
- [Screenshots](#screenshots)
- [Installation](#installation)
  - [Quick Setup with uv](#quick-setup-with-uv)
  - [Alternative Setup with pip](#alternative-setup-with-pip)
- [Usage](#usage)
  - [Getting Started](#getting-started)
  - [Settings Configuration](#settings-configuration)
  - [Camera Positioning](#camera-positioning)
- [How It Works](#how-it-works)
  - [Posture Detection](#posture-detection)
  - [Alert System](#alert-system)
  - [Technical Architecture](#technical-architecture)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Building Executable](#building-executable)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**Posture Guard** (also known as "Fixture") is a modern PyQt6 desktop application that monitors your posture in real-time using your computer's camera. The app runs entirely on-device, analyzing your posture from a side camera view and providing instant visual and audio alerts when slouching is detected.

Perfect for office workers, gamers, students, and anyone who spends extended periods at their desk.

## The Problem

Poor posture is a widespread issue affecting millions of people:

- **Long work hours** lead to slouching and forward head posture
- **Gaming sessions** cause neck and back strain
- **Focused work** makes us forget proper sitting posture
- **Chronic effects** include neck pain, back problems, headaches, and long-term health issues

## The Solution

Posture Guard provides continuous, intelligent monitoring that:

- ✅ Watches your posture in real-time using AI-powered pose detection
- ✅ Alerts you instantly when you start to slouch
- ✅ Uses visual warnings and voice reminders to keep you mindful
- ✅ Works completely offline on your device
- ✅ Protects your privacy - no data leaves your computer

---

## Features

### Core Functionality

- **Real-time Posture Monitoring** - Uses MediaPipe pose detection to analyze your posture continuously
- **Visual Alerts** - Red overlay warnings when bad posture is detected
- **Voice Notifications** - High-quality Google TTS voice reminders (customizable delay)
- **Multiple Camera Support** - Automatic camera detection with easy switching
- **Adjustable Sensitivity** - Fine-tune detection thresholds to match your needs
- **Modern UI** - Clean, dark-themed PyQt6 interface with settings modal

### Privacy & Security

- **100% Offline Processing** - All pose detection runs locally on your device
- **No Data Collection** - Your video feed never leaves your computer
- **No Cloud Dependencies** - Complete functionality without internet (except TTS generation)

### Customization

- **Sensitivity Control** - Adjust posture detection threshold (10% - 90%)
- **Voice Trigger Delay** - Set how long to wait before voice alert (1-30 seconds)
- **Toggle Alerts** - Enable/disable visual and voice alerts independently
- **Camera Selection** - Choose from multiple detected cameras

---

## Screenshots

_Coming soon - add screenshots of your application in action_

---

## Installation

### Prerequisites

- **Python 3.10 or higher**
- **Webcam or external camera** (positioned to view you from the side)
- **macOS, Windows, or Linux**

### Quick Setup with uv

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

1. **Install uv** (if not already installed):

   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/posture-app.git
   cd posture-app
   ```

3. **Install dependencies**:

   ```bash
   uv sync
   ```

4. **Run the application**:
   ```bash
   uv run python posture_guard.py
   ```

### Alternative Setup with pip

If you prefer using pip:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/posture-app.git
   cd posture-app
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv .venv

   # Activate the virtual environment
   # On macOS/Linux:
   source .venv/bin/activate

   # On Windows:
   .venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install PyQt6 opencv-python mediapipe numpy gtts pygame
   ```

4. **Run the application**:
   ```bash
   python posture_guard.py
   ```

---

## Usage

### Getting Started

1. **Position Your Camera**

   - Place your camera to view you from the **side** (not front-facing)
   - Ensure your ear, shoulder, and hip are visible
   - See [Camera Positioning](#camera-positioning) for detailed guidance

2. **Launch the Application**

   ```bash
   uv run python posture_guard.py
   ```

3. **Start Monitoring**

   - Click **"Start Camera"** button
   - Sit with good posture initially
   - The app will begin monitoring automatically

4. **Receive Alerts**
   - **Green status** = Good posture ✅
   - **Red status** = Bad posture detected ❌
   - Visual overlay appears on bad posture
   - Voice reminder after configured delay

### Settings Configuration

Click the **⚙️ Settings** button to customize:

#### Camera Settings

- **Select Camera** - Choose from automatically detected cameras
- Cameras are tested and listed as "Camera 0 (default)", "Camera 1", etc.

#### Alert Settings

- **Sensitivity** (10% - 90%)

  - Lower values = more sensitive (triggers alerts more easily)
  - Higher values = less sensitive (only alerts on severe slouching)
  - Default: 30%

- **Voice Trigger Delay** (1.0 - 30.0 seconds)

  - Time in bad posture before voice alert triggers
  - Default: 3.0 seconds
  - Prevents constant alerts for minor adjustments

- **Visual Alerts** - Toggle red overlay warnings
- **Voice Alerts** - Toggle spoken reminders

### Camera Positioning

For optimal posture detection:

```
           Camera
             |
             v
          [Side View]

    Ear •
         \
          • Shoulder
           \
            \
             • Hip
```

**Best Practices:**

- Position camera at shoulder height or slightly above
- Ensure side profile is clearly visible
- Maintain consistent distance from camera
- Good lighting helps detection accuracy
- All three landmarks (ear, shoulder, hip) should be visible

---

## How It Works

### Posture Detection

Posture Guard uses **MediaPipe Pose Detection**, a state-of-the-art machine learning model that:

1. **Detects Key Landmarks**

   - Right ear (landmark 8)
   - Right shoulder (landmark 12)
   - Right hip (landmark 24)

2. **Calculates Posture Metrics**

   - **Shoulder Angle**: Angle between ear-shoulder and hip-shoulder vectors
   - **Forward Head Position**: Horizontal distance between ear and shoulder
   - **Combined Score**: Weighted combination of both metrics

3. **Applies Posture Guidelines**

   - **Neutral alignment**: 165° - 180° (Good posture)
   - **Mild forward head**: 150° - 165° (Borderline)
   - **Bad posture**: < 150° (Triggers alerts)

4. **Smoothing & Stability**
   - 5-frame smoothing buffer reduces false positives
   - Time-based accumulation prevents alert spam
   - Configurable thresholds for personalization

### Alert System

#### Visual Alerts

- Red status chip in UI
- Overlay guidance lines on video feed
- Real-time angle and confidence display
- Detailed logging in activity panel

#### Voice Alerts

- **High-quality Google TTS** (gTTS) for natural speech
- Cached audio playback via pygame
- **Time-based triggering**: Only speaks after sustained bad posture
- **Cooldown period**: 4 seconds between voice alerts
- **Configurable delay**: 1-30 seconds (default: 3s)

#### Smart State Management

- Tracks bad posture duration
- Resets timer when good posture is restored
- Prevents alert fatigue with intelligent throttling
- Debug output every 30 frames for monitoring

### Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Posture Guard App                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐      ┌─────────────────────────────┐  │
│  │  PyQt6 UI    │◄─────┤   Video Thread (QThread)    │  │
│  │              │      │  - Camera capture           │  │
│  │  - Video     │      │  - Frame processing         │  │
│  │  - Status    │      │  - Backend inference        │  │
│  │  - Settings  │      │  - Smoothing buffer         │  │
│  │  - Logs      │      └───────────┬─────────────────┘  │
│  └──────────────┘                  │                    │
│         │                          │                    │
│         │                          v                    │
│  ┌──────▼──────────────────────────────────────────┐    │
│  │        HeuristicPoseBackend                     │    │
│  │  - MediaPipe Pose Detection                     │    │
│  │  - Angle calculation                            │    │
│  │  - Posture classification                       │    │
│  │  - Landmark visualization                       │    │
│  └─────────────────────────────────────────────────┘    │
│         │                                               │
│         │                                               │
│  ┌──────▼───────────────────────────────────────────┐   │
│  │        Speech System (Threading)                 │   │
│  │  - Google TTS (gTTS)                             │   │
│  │  - Audio queue management                        │   │
│  │  - Pygame mixer playback                         │   │
│  │  - Cooldown enforcement                          │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **Video Thread** (`VideoThread`)

   - Runs in separate QThread for non-blocking UI
   - Manages camera lifecycle and frame capture
   - Handles camera switching while running
   - Applies smoothing to detections

2. **Pose Backend** (`HeuristicPoseBackend`)

   - MediaPipe-based pose estimation
   - Multi-metric posture analysis
   - Configurable thresholds
   - Visual overlay rendering

3. **Speech System**

   - Threaded speech queue
   - Google TTS for natural voice
   - Pygame mixer for audio playback
   - Temporary file management

4. **Settings Dialog** (`SettingsDialog`)
   - Dynamic camera detection
   - Real-time setting updates
   - Save/cancel with rollback support

---

## Project Structure

```
posture-app/
├── README.md                 # This file
├── PRESENTATION.md           # Project presentation slides
├── pyproject.toml            # Project configuration and dependencies
├── uv.lock                   # Locked dependency versions
├── .python-version           # Python version specification (3.10)
├── .gitignore                # Git ignore rules
│
├── posture_guard.py          # Main application entry point
│
├── src/                      # Modular source code (optional architecture)
│   ├── backend/              # Detection backends
│   ├── core/                 # Core utilities (speech, video)
│   ├── config/               # Configuration management
│   └── ui/                   # UI components
│
├── test_model.py             # YOLO model testing script
├── test_backend.py           # Backend validation script
│
├── yolov3-obj.cfg            # YOLOv3 configuration (optional backend)
├── posture.names             # Class labels for YOLO (optional)
│
└── docs/                     # Documentation (empty, for future use)
```

### Key Files

- **posture_guard.py** - Main application with all functionality
- **pyproject.toml** - Project metadata and dependencies
- **uv.lock** - Reproducible dependency versions
- **yolov3-obj.cfg** - Configuration for optional YOLO backend
- **test_model.py** - Utility to test YOLO model loading

---

## Configuration

### Posture Detection Parameters

You can adjust these in the code (`posture_guard.py`):

```python
# In HeuristicPoseBackend.__init__()
self.angle_threshold = 15.0  # Degrees for angle-based detection
self.smoothing_buffer = deque(maxlen=5)  # Frames for smoothing

# In PostureApp.__init__()
self.min_bad_ratio = 0.3  # Threshold for bad posture (0.0 - 1.0)
self.voice_trigger_delay = 3.0  # Seconds before voice alert
self.speech_cooldown = 4.0  # Seconds between voice alerts
```

### Camera Settings

```python
# In VideoThread._init_camera()
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

---

## Building Executable

Create a standalone executable for distribution:

### Using PyInstaller with uv

```bash
# Install PyInstaller as dev dependency
uv add --dev pyinstaller

# Build executable
uv run pyinstaller \
    --noconfirm \
    --name PostureGuard \
    --onefile \
    --windowed \
    posture_guard.py

# Executable will be in dist/PostureGuard
```

### Platform-Specific Notes

**macOS:**

```bash
# May need to sign the app for distribution
codesign --force --deep --sign - dist/PostureGuard.app
```

**Windows:**

```bash
# Creates PostureGuard.exe in dist/
# May trigger Windows Defender - add exception if needed
```

**Linux:**

```bash
# Creates PostureGuard binary in dist/
# Make executable: chmod +x dist/PostureGuard
```

---

## Troubleshooting

### Camera Issues

**Problem:** Camera not detected or "No cameras detected" message

**Solutions:**

- Ensure camera is not being used by another application
- Check system camera permissions (Settings > Privacy > Camera)
- Try different camera indices in Settings (0, 1, 2, etc.)
- Restart the application after connecting a new camera

**macOS specific:**

```bash
# Grant terminal camera access if running from command line
# System Settings > Privacy & Security > Camera > Terminal
```

### MediaPipe Issues

**Problem:** MediaPipe import errors or detection failures

**Solutions:**

```bash
# Reinstall MediaPipe
uv remove mediapipe
uv add mediapipe

# Or with pip
pip uninstall mediapipe
pip install mediapipe
```

**Platform support:**

- MediaPipe supports macOS, Windows, and Linux
- May require additional system libraries on some Linux distributions:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev
  ```

### Voice/TTS Issues

**Problem:** Voice alerts not working

**Solutions:**

1. **Check TTS availability:**

   - Look for "Voice alerts enabled with high-quality TTS" in console
   - If not present, check pygame and gTTS installation

2. **Reinstall speech dependencies:**

   ```bash
   uv add gtts pygame
   # or
   pip install gtts pygame
   ```

3. **Internet requirement:**

   - gTTS requires internet connection for speech generation
   - Audio is cached locally after first generation
   - Check firewall settings if blocked

4. **Audio playback issues:**
   ```bash
   # Test pygame audio
   python -c "import pygame; pygame.mixer.init(); print('OK')"
   ```

### Performance Issues

**Problem:** Laggy video or high CPU usage

**Solutions:**

- Lower camera resolution in code (640x480 instead of 1280x720)
- Increase frame skip in video processing
- Close other camera-using applications
- Check CPU usage - MediaPipe is CPU-intensive

### Detection Issues

**Problem:** Posture not detected correctly or too many false alerts

**Solutions:**

1. **Adjust sensitivity:**

   - Settings > Sensitivity slider
   - Lower sensitivity (30-40%) for less strict detection
   - Higher sensitivity (60-70%) for stricter detection

2. **Improve camera positioning:**

   - Ensure side profile is clearly visible
   - Ear, shoulder, and hip must all be in frame
   - Consistent distance from camera
   - Good lighting conditions

3. **Check debug output:**
   - Console shows angle measurements and detection status
   - Neutral posture should show 165-180°
   - Bad posture typically shows < 150°

---

## Development

### Running Tests

```bash
# Test YOLO model loading (if using YOLO backend)
uv run python test_model.py

# Test backend inference
uv run python test_backend.py
```

### Development Setup

```bash
# Install with dev dependencies
uv sync --dev

# Run in development mode with verbose output
uv run python posture_guard.py
```

### Code Style

This project follows:

- **PEP 8** style guidelines
- **Type hints** where applicable
- **Docstrings** for classes and complex functions

### Architecture Notes

- **Backends**: The app supports multiple detection backends (ONNX, YOLO, MediaPipe)
- **Current default**: MediaPipe Heuristic backend (no model files needed)
- **Extension**: Easy to add new backends by inheriting from `BaseBackend`

---

## Dependencies

### Core Dependencies

| Package           | Version  | Purpose                                |
| ----------------- | -------- | -------------------------------------- |
| **PyQt6**         | 6.9.1+   | Modern GUI framework                   |
| **opencv-python** | 4.11.0+  | Camera capture and image processing    |
| **mediapipe**     | 0.10.21+ | Pose detection and landmark estimation |
| **numpy**         | 1.26.4+  | Numerical computations                 |
| **gtts**          | 2.5.1+   | Google Text-to-Speech for voice alerts |
| **pygame**        | 2.5.2+   | Audio playback for TTS                 |

### Optional Dependencies

| Package         | Purpose                         |
| --------------- | ------------------------------- |
| **onnxruntime** | ONNX model inference backend    |
| **ultralytics** | YOLOv8 detection backend        |
| **pyinstaller** | Building standalone executables |

### System Requirements

- **Python**: 3.10 or higher
- **RAM**: 2GB minimum, 4GB recommended
- **Disk**: 500MB for dependencies
- **Camera**: Any webcam or external camera
- **OS**: macOS 10.15+, Windows 10+, or Linux (Ubuntu 20.04+)

---

## Contributing

Contributions are welcome! This project is designed to be simple and maintainable.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: description"
   ```
6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

### Areas for Contribution

- **Additional pose detection backends**
- **Improved UI/UX design**
- **Cross-platform testing and fixes**
- **Performance optimizations**
- **Documentation improvements**
- **Localization/translations**
- **Unit tests and integration tests**

### Code Guidelines

- Follow existing code structure and style
- Add docstrings for new functions/classes
- Test on your platform before submitting PR
- Keep dependencies minimal
- Maintain privacy-first approach (no data collection)

---

## License

This project is open source. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **MediaPipe** by Google - Pose detection technology
- **PyQt6** - GUI framework
- **OpenCV** - Computer vision library
- **gTTS** - Text-to-speech engine

---

## Support

For issues, questions, or suggestions:

1. **Check [Troubleshooting](#troubleshooting)** section
2. **Search existing [GitHub Issues](https://github.com/yourusername/posture-app/issues)**
3. **Open a new issue** with detailed description

---

## Roadmap

Future enhancements being considered:

- [ ] Multi-user profile support
- [ ] Posture statistics and history tracking
- [ ] Customizable alert sounds and messages
- [ ] Break reminders and stretch suggestions
- [ ] Machine learning model training tools
- [ ] Mobile companion app
- [ ] Cloud sync for multi-device (optional, privacy-preserving)

---

<div align="center">

**Stay healthy. Sit straight.**

Made with ❤️ for better posture

[⬆ Back to Top](#posture-guard)

</div>
