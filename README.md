# Posture Guard — Side View Desktop App

A modern PyQt6 desktop application that runs entirely on-device to monitor your posture from a side camera view. The app uses MediaPipe pose detection to analyze your posture in real-time and provides visual and audio alerts when bad posture is detected.

## Features

- **Real-time posture monitoring** using MediaPipe pose detection
- **Visual alerts** with overlay warnings for bad posture
- **Voice notifications** that trigger on each bad posture detection
- **Multiple camera support** with camera selector
- **Adjustable sensitivity** for posture detection
- **Modern UI** with settings modal
- **Completely offline** - no data is sent anywhere

## Quick Setup with uv

This project uses `uv` for fast Python package management. Make sure you have [uv installed](https://github.com/astral-sh/uv).

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Posture-App
```

2. Install dependencies with uv:
```bash
uv sync
```

3. Run the application:
```bash
uv run python posture_guard.py
```

## Alternative Setup (without uv)

If you prefer using pip:

1. Create a Python 3.10+ virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python posture_guard.py
```

## Usage

1. **Start the camera**: Click "Start Camera" to begin monitoring
2. **Adjust settings**: Click the "⚙️ Settings" button to:
   - Select your camera (if you have multiple cameras)
   - Adjust posture detection sensitivity
   - Enable/disable visual alerts
   - Enable/disable voice notifications
3. **Position yourself**: Sit side-on to your camera for best results
4. **Monitor your posture**: The app will show live video with posture analysis
   - Green status = Good posture
   - Red status = Bad posture (triggers alerts)

## How It Works

The application uses MediaPipe's pose detection to analyze the angle between your ear, shoulder, and hip. When this angle indicates forward head posture or slouching, the app triggers alerts.

### Posture Detection
- Uses MediaPipe Pose landmarks (ear, shoulder, hip)
- Calculates shoulder angle for posture assessment
- Configurable sensitivity threshold
- Real-time analysis at camera frame rate

### Alert System
- **Visual alerts**: Red overlay with "Bad posture" message
- **Voice alerts**: Spoken reminder to straighten up
- **State-based triggering**: Alerts only when transitioning from good to bad posture

## Project Structure

```
Posture-App/
├── posture_guard.py     # Main application file
├── pyproject.toml       # Project configuration and dependencies
├── uv.lock             # Locked dependency versions
├── README.md           # This file
└── .python-version     # Python version specification
```

## Technical Details

- **Framework**: PyQt6 for the GUI
- **Computer Vision**: MediaPipe for pose detection
- **Image Processing**: OpenCV for camera handling
- **Text-to-Speech**: pyttsx3 for voice alerts
- **Python Version**: 3.10+

## Building Executable (Optional)

To create a standalone executable:

```bash
# Install PyInstaller
uv add --dev pyinstaller

# Build executable
uv run pyinstaller --noconfirm --name PostureGuard --onefile --windowed posture_guard.py
```

The executable will be created in the `dist/` directory and runs completely offline.

## Troubleshooting

### Camera Issues
- Make sure your camera is not being used by another application
- Try different camera indices in Settings if the default doesn't work
- Ensure you have camera permissions on your system

### MediaPipe Issues
- MediaPipe requires a supported platform
- On some systems, you may need to install additional system libraries

### Voice Issues
- The app uses Google Text-to-Speech for high-quality voice alerts
- Requires internet connection for speech generation (audio is cached locally)
- If you experience audio issues, ensure pygame and system audio are working

## Dependencies

- **PyQt6**: Modern GUI framework
- **OpenCV**: Camera and image processing
- **MediaPipe**: Pose detection and analysis
- **NumPy**: Numerical computations
- **gTTS**: Google Text-to-Speech for high-quality voice alerts
- **pygame**: Audio playback for speech system

## Contributing

This is a focused posture monitoring application. The codebase is designed to be simple and maintainable while providing effective posture monitoring functionality.

## License

This project is open source. Check the license file for details.