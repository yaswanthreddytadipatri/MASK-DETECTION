# MASK_DETECTION

## Overview
This repository contains a Python script for face mask detection using a pre-trained model. It utilizes the MobileNetV2 architecture to detect faces and predict whether a person is wearing a mask or not.

## Main Python File
- **Face_Mask_Detection.py**: This is the main Python script that performs face mask detection. It uses OpenCV, TensorFlow, and the MobileNetV2 model for this task.

## Installation
1. Clone the repository to your local machine:
   
   ```bash
   git clone https://github.com/yourusername/MASK_DETECTION.git

2. Navigate to the project directory:
   ```bash
   cd MASK_DETECTION

3. Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

## How to Use
1. Make sure you have completed the installation steps mentioned above.

2. Run the `Face_Mask_Detection.py` script:
   ```bash
   python Face_Mask_Detection.py
   ```

3. The script will start a video stream and detect faces with or without masks, displaying the result in real-time.

4. Press 'q' to quit the video stream.
