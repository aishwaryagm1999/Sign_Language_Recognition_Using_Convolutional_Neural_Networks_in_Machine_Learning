ASLVizNet
Real-Time American Sign Language Recognition using CNN & TensorFlow Object Detection API
📌 Overview

ASLVizNet is a real-time computer vision framework designed for recognizing static American Sign Language (ASL) alphabets and numbers using deep convolutional neural networks and transfer learning.

The system leverages the TensorFlow Object Detection API and SSD MobileNet v2 to perform bounding box localization and classification of hand gestures from live webcam input.

ASLVizNet was developed as a research-driven project and presented at the IACIT 2021 Conference, with publication in IJARCS (International Journal of Advanced Research in Computer Science).

🎯 Problem Statement

Traditional sign language translation systems:

Depend on expensive sensor gloves

Require specialized hardware

Lack real-time responsiveness

Provide limited accessibility

ASLVizNet proposes a low-cost, vision-based deep learning approach that:

Uses only a webcam

Performs real-time detection

Achieves high accuracy (96–99%)

Requires no wearable devices

🏗️ System Architecture
Webcam Input (OpenCV)
        ↓
Image Annotation (LabelImg - XML)
        ↓
XML → TFRecord Conversion
        ↓
TensorFlow Object Detection API
        ↓
SSD MobileNet v2 (Transfer Learning)
        ↓
Real-Time Detection with Bounding Box + Confidence Score
🧠 Deep Learning Methodology
🔹 Model Architecture

Model: SSD MobileNet v2

Framework: TensorFlow Object Detection API

Approach: Transfer Learning

Detection Type: Object Detection (Bounding Box + Classification)

🔹 Why SSD MobileNet v2?

Lightweight architecture

Optimized for real-time inference

Efficient for low-compute environments

Strong balance between speed and accuracy

📂 Dataset Pipeline
1️⃣ Data Collection

Custom ASL gesture dataset created

Static alphabets (A–Z)

Numbers (0–9)

Images captured using OpenCV

2️⃣ Annotation

Tool: LabelImg

Generated XML annotation files

Bounding box coordinates labeled per image

3️⃣ TFRecord Generation

Used custom script:

generate_tfrecord.py

This script:

Parses XML files

Converts annotations to TFRecord format

Maps labels using .pbtxt

Optionally generates CSV file

🔬 Model Training
Training Configuration

Framework: TensorFlow

Training Steps: 10,000 epochs

Final Training Loss: 0.086

Hardware Used

Intel i5 Processor

8GB RAM

GTX 1030 (Optional GPU Acceleration)

Webcam for live testing

Training Process

Cloned TensorFlow Model Zoo

Selected SSD MobileNet v2 configuration

Modified pipeline.config

Generated label map (.pbtxt)

Converted dataset to TFRecord

Trained model

Exported trained model for inference

📊 Experimental Results
Metric	Value
Training Epochs	10,000
Final Loss	0.086
Real-Time Accuracy	96% – 99%
Detection Output	Bounding Box + Confidence Score
Input Device	Webcam

The system successfully performs real-time gesture detection with high confidence prediction scores.

🛠️ Technologies Used
🔹 Programming

Python 3.x

🔹 Computer Vision

OpenCV

NumPy

Pillow

🔹 Deep Learning

TensorFlow

TensorFlow Object Detection API

SSD MobileNet v2

Transfer Learning

TFRecord format

Label Map (.pbtxt)

🔹 Data Processing

Pandas

XML parsing (ElementTree)

TFRecord serialization

▶️ Steps to Reproduce
1️⃣ Clone Repository
git clone https://github.com/yourusername/ASLVizNet.git
cd ASLVizNet
2️⃣ Install Dependencies
pip install tensorflow opencv-python pandas numpy pillow lxml

Install TensorFlow Object Detection API dependencies.

3️⃣ Annotate Dataset

Capture gesture images

Annotate using LabelImg

Save XML files in /annotations

4️⃣ Generate TFRecords
python generate_tfrecord.py \
-x annotations \
-l label_map.pbtxt \
-o train.record \
-i images
5️⃣ Train Model
python model_main_tf2.py \
--pipeline_config_path=training/pipeline.config \
--model_dir=training/ \
--alsologtostderr
6️⃣ Run Real-Time Detection
python real_time_detection.py

Webcam will activate and display:

Bounding box

Predicted ASL character

Confidence score

📚 Research Publication

Presented at:

IACIT 2021 Conference

Published in:

International Journal of Advanced Research in Computer Science (IJARCS)

“Sign Language Recognition using Convolutional Neural Networks in Machine Learning”, IJARCS, Vol. 12, pp. 16–20, Aug. 2021.
DOI: 10.26483/ijarcs.v12i0.6713

🎓 Skills Demonstrated

Computer Vision

Deep Learning

TensorFlow Ecosystem

Transfer Learning

Dataset Engineering

TFRecord Pipeline Development

Real-Time ML Deployment

Research Publication & Presentation
