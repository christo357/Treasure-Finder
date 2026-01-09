# Treasure Finder 🎯

A real-time computer vision game that uses YOLO object detection to create an interactive treasure hunt experience through your webcam.

## Overview

Treasure Finder is an interactive game that randomly selects an object from your environment as "treasure" and challenges you to find it. Using YOLOv8/v11 models and object tracking, the application provides real-time feedback and scoring based on how quickly you locate the treasure.

## Features

- **Real-time Object Detection**: Using YOLO (YOLOv8/v11) models
- **Object Tracking**: Uses DeepSort for robust multi-object tracking
- **Interactive UI**: Built with Streamlit for an intuitive web interface
- **Scoring System**: Time-based scoring that rewards quick discoveries
- **Webcam Integration**: Works with any standard webcam

## Demo

https://github.com/user-attachments/assets/385158c8-9142-46cc-95f3-3f972f39fb8f




## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Treasure-Finder.git
cd Treasure-Finder
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

Or run the standalone version:
```bash
python Treasure.py
```

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLO
- Streamlit
- Deep Sort Realtime
- See [requirements.txt](requirements.txt) for full list

## How It Works

1. The application starts your webcam feed
2. YOLO detects all objects in the frame (excluding people)
3. A random object is selected as the "treasure"
4. Find and center the treasure object in your camera view
5. Score points based on how quickly you find it!

## Project Structure

- `app.py` - Main Streamlit application
- `Treasure.py` - Standalone game implementation
- `capture.py` - Camera capture utilities
- `demo.py` - Demo/testing script
- `roboflow.py` - Roboflow integration
- `yolo11n.pt`, `yolov8n.pt` - Pre-trained YOLO models

## License

MIT License
