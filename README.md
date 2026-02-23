# ASLVizNet  
### Real-Time American Sign Language Recognition using CNN & TensorFlow Object Detection API

---

## 📌 Overview

**ASLVizNet** is a real-time computer vision framework designed for recognizing static American Sign Language (ASL) alphabets and numbers using deep convolutional neural networks and transfer learning.

The system leverages the **TensorFlow Object Detection API** and **SSD MobileNet v2** to perform bounding box localization and classification of hand gestures from live webcam input.

ASLVizNet was developed as a research-driven project and presented at the **IACIT 2021 Conference**, with publication in the **International Journal of Advanced Research in Computer Science (IJARCS)**.

---

## 🎯 Problem Statement

Traditional sign language translation systems:

- Depend on expensive sensor gloves  
- Require specialized hardware  
- Lack real-time responsiveness  
- Provide limited accessibility  

ASLVizNet proposes a low-cost, vision-based deep learning approach that:

- Uses only a webcam  
- Performs real-time detection  
- Achieves high accuracy (96–99%)  
- Requires no wearable devices  

---

## 🏗️ System Architecture

```text
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
```

---

## 🧠 Deep Learning Methodology

### 🔹 Model Architecture

- **Model:** SSD MobileNet v2  
- **Framework:** TensorFlow Object Detection API  
- **Approach:** Transfer Learning  
- **Detection Type:** Object Detection (Bounding Box + Classification)  

### 🔹 Why SSD MobileNet v2?

- Lightweight architecture  
- Optimized for real-time inference  
- Efficient for low-compute environments  
- Strong balance between speed and accuracy  

---

## 📂 Dataset

The complete dataset (gesture images + annotations) is available here:

🔗 **Google Drive Dataset Link**  
https://drive.google.com/drive/folders/1_vZt3Jn-JPQU5viHmGyGMdwQshuFqZOT?usp=sharing  

The dataset contains:

- ASL Alphabets (A–Z)
- Numbers (0–9)
- XML annotation files (LabelImg format)
- Images used for training

⚠️ Note: Dataset is hosted externally due to GitHub size limitations.

---

## 📁 Project Structure

```
ASLVizNet/
│
├── annotations/              # XML files from LabelImg
├── images/                   # Gesture images
├── training/                 # Model checkpoints
├── exported-model/           # Final exported model
│
├── ImageCapture.ipynb        # Dataset capture notebook
├── MainCode.ipynb            # Real-time detection notebook
├── generate_tfrecord.py      # XML → TFRecord converter
├── label_map.pbtxt           # Class label definitions
├── pipeline.config           # Training configuration
└── README.md
```

---

## 🛠️ Requirements

### 🔹 Tested Environment

- Python 3.7  
- TensorFlow 2.4.1  
- CUDA (Optional for GPU acceleration)

### 🔹 Required Libraries

```bash
pip install tensorflow==2.4.1
pip install opencv-python
pip install pandas numpy pillow lxml
```

---

## 🔧 Install TensorFlow Object Detection API

```bash
git clone https://github.com/tensorflow/models.git
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
pip install .
```

---

## 📂 Dataset Preparation

1. Download dataset from Google Drive.
2. Place images inside `/images`
3. Place XML files inside `/annotations`
4. Ensure `label_map.pbtxt` contains correct class mappings.

---

## 🔁 Generate TFRecords

```bash
python generate_tfrecord.py \
-x annotations \
-l label_map.pbtxt \
-o train.record \
-i images
```

This script:

- Parses XML files  
- Converts annotations to TFRecord format  
- Maps labels using `.pbtxt`  
- Optionally generates CSV file  

---

## 🔬 Model Training

```bash
python model_main_tf2.py \
--pipeline_config_path=training/pipeline.config \
--model_dir=training/ \
--alsologtostderr
```

### Training Configuration

- Training Steps: 10,000 epochs  
- Final Training Loss: 0.086  
- Real-Time Accuracy: 96–99%  

---

## 📦 Export Trained Model

```bash
python exporter_main_v2.py \
--input_type image_tensor \
--pipeline_config_path training/pipeline.config \
--trained_checkpoint_dir training/ \
--output_directory exported-model
```

---

## ▶️ Run Real-Time Detection

```bash
python MainCode.ipynb
```

OR open notebook and run all cells.

Webcam will activate and display:

- Bounding box  
- Predicted ASL character  
- Confidence score  

---

## 📊 Experimental Results

| Metric | Value |
|--------|--------|
| Training Epochs | 10,000 |
| Final Loss | 0.086 |
| Real-Time Accuracy | 96% – 99% |
| Detection Output | Bounding Box + Confidence Score |

The system successfully performs real-time gesture detection with high confidence prediction scores.

---

## 📚 Research Publication

Presented at:

**IACIT 2021 Conference**

Published in:

**International Journal of Advanced Research in Computer Science (IJARCS)**  

> “Sign Language Recognition using Convolutional Neural Networks in Machine Learning”, IJARCS, Vol. 12, pp. 16–20, Aug. 2021.  
> DOI: 10.26483/ijarcs.v12i0.6713  

---


---

## 🎓 Skills Demonstrated

- Computer Vision  
- Deep Learning  
- TensorFlow Object Detection API  
- Transfer Learning  
- Dataset Engineering  
- TFRecord Pipeline Development  
- Real-Time ML Deployment  
- Research Publication & Presentation  

---

## ⚠️ Note on Large Files

Model checkpoints and dataset files are not included in the repository due to GitHub size limits. Please use the provided dataset link and training instructions to reproduce results.

---

## 👩‍💻 Author

Developed as a research-driven computer vision framework integrating deep learning and real-time detection for assistive communication systems.
