# Air Canvas

**Air Pen** is an innovative tool that allows you to draw on a digital canvas simply by capturing the motion of a colored marker with a camera. This project leverages computer vision techniques to track and interpret the movement of a colored object at the tip of your finger, enabling you to create drawings in the air.

## Project Overview

The core of the Air Pen project is built using **OpenCV** in Python, taking advantage of its powerful libraries and user-friendly syntax. The project also utilizes **Mediapipe** for hand detection, specifically using Mediapipe's hand detection model to track finger movements. However, the principles and techniques used can be adapted to any programming language that supports OpenCV.

## Key Features

- **Color Detection and Tracking**: Utilizes advanced color detection algorithms to identify the marker's color and track its movement in real-time.
- **Hand Detection**: Uses Mediapipe's hand detection model to track finger movements and interpret gestures.
- **Mask Creation**: Generates a binary mask based on the detected color, isolating the marker from the background.
- **Morphological Operations**: Applies Erosion and Dilation to refine the mask quality.
  - **Erosion**: Reduces noise and impurities in the mask by eroding the boundaries of the detected color region.
  - **Dilation**: Restores the main mask area after erosion to ensure the primary drawing area remains intact.

## Compatibility

- **Mediapipe**: Currently supports Python versions 3.8 to 3.10. Ensure you are using one of these versions for compatibility with the Mediapipe library.

## How It Works

1. **Capture**: A camera captures the movement of a colored marker.
2. **Hand Detection**: Mediapipe's hand detection model identifies and tracks finger movements.
3. **Color Detection**: The system identifies the color of the marker.
4. **Mask Creation**: A binary mask is created to highlight the detected color.
5. **Morphological Processing**: Erosion and dilation refine the mask for accurate tracking.
6. **Drawing**: The processed mask is used to draw on a digital canvas in real-time.

## Getting Started

1. **Install Dependencies**: Ensure you have Python 3.8 to 3.10 and OpenCV installed. You can install OpenCV using pip:
   ```bash
   pip install opencv-python
2. **Install  Mediapie**:
   ```bash
   pip install mediapipe
3. Run the code
   ```bash
   python3 ml.py
4. **Enjoy Drawing**: Use the colored marker to draw in the air and see your creations come to life on the screen!
