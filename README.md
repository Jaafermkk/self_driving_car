# Object Detection for Self-Driving Cars using YOLOv3 and Keras

This project explores object detection for self-driving car applications using the YOLOv3 (You Only Look Once) algorithm implemented in Keras.  The goal is to identify and locate objects like cars, pedestrians, and traffic lights in images captured by a self-driving car's camera.

## Project Overview

This notebook demonstrates the initial steps of setting up an object detection pipeline.  It focuses on data preparation and leverages pre-trained YOLOv3 weights.  The code includes:

* Mounting Google Drive to access datasets.
* Downloading and extracting training and testing image datasets (currently encountering errors).
* Basic file system operations for data management.
* Partial implementation of the YOLOv3 model architecture in Keras.

## Dataset

The project intends to use a dataset of images captured from a self-driving car's perspective.  The dataset is expected to contain labeled bounding boxes for relevant objects. The notebook includes code to download the dataset from Kaggle, but the provided URLs are currently expired, resulting in 400 Bad Request errors.  Replacement or alternative data sources will be necessary to run the notebook successfully.

## Model

The project utilizes the YOLOv3 architecture, known for its real-time object detection capabilities.  The implementation is based on Keras. The provided notebook only includes the initial setup and data handling portions. The actual model training and evaluation parts are missing.

## Dependencies

The project relies on the following libraries:

* TensorFlow/Keras
* OpenCV (cv2)
* NumPy
* Other standard Python libraries (os, argparse, struct)

## Usage

1. Data Acquisition:  Acquire a suitable dataset for object detection in self-driving car scenarios.  The current download links in the notebook are broken.
2. Data Preparation:  Organize the dataset into appropriate training and testing sets.  The notebook provides a starting point for this process.
3. Model Implementation: Complete the YOLOv3 model implementation in Keras.  Refer to the provided resources for guidance.
4. Training: Train the YOLOv3 model on the prepared dataset.
5. Evaluation: Evaluate the trained model's performance on the test set using appropriate metrics (e.g., mean average precision).

## Challenges and Future Work

* Broken Download Links: The current links to download the dataset are not functional.  Finding a replacement dataset is a priority.
* Incomplete Model Implementation: The YOLOv3 model implementation is incomplete.  The core model architecture, training loop, and evaluation metrics need to be added.
* Performance Optimization:  Explore techniques for optimizing the model's performance for real-time object detection on resource-constrained devices.


## References

* [YOLOv3 Keras 2D Object Detection Kaggle Notebook](https://www.kaggle.com/code/sakshaymahna/yolov3-keras-2d-object-detection/notebook)
* [How to Perform Object Detection With YOLOv3 in Keras](https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/)
