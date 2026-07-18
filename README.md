CodeAlpha Artificial Intelligence Internship

This repository contains my submissions for the CodeAlpha Artificial Intelligence Internship program. As per the program guidelines, I have completed the following selected projects:

Task 1: Language Translation Tool

Task 4: Real-Time Object Detection & Tracking

 Selected Tasks Overview

1. Task 1: Language Translation Tool

A user-friendly translator application capable of converting text from a source language to a selected target language.

Input Interface: A clean user interface allowing users to enter custom text and select their desired source/target languages.

Translation Engine: Powered by machine translation models via translation APIs (e.g., Google Translate API or Microsoft Translator).

Features: Instant text translation rendering, supporting smooth multilingual communication.

2. Task 4: Real-Time Object Detection & Tracking

A computer vision pipeline that processes live feeds to locate, identify, and persistently track objects across consecutive video frames.

Frame Acquisition: Capturing live frames from a webcam feed or custom video stream using OpenCV.

Core Detection: Using pre-trained deep learning networks (such as YOLO or Faster R-CNN) to detect objects and outline them with bounding boxes.

Object Tracking: Implementing real-time tracking algorithms (such as SORT or Deep SORT) to match detections across frames, generating persistent IDs and tracking trails for every unique object.




📌 Getting Started

Prerequisites

Make sure you have Python installed on your local machine.

Task 1: Run the Translator

Navigate to the task directory:

cd Task_1_Language_Translation


Install dependencies:

pip install -r requirements.txt


Run the application:

python app.py


Task 4: Run the Object Tracker

Navigate to the task directory:

cd Task_4_Object_Detection


Install dependencies:

pip install -r requirements.txt


Run the tracking script:

python main.py
