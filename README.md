# Pixel Persona (Gender and Age Detection using CNN)

This project uses **Convolutional Neural Networks (CNNs)** to predict a person’s **gender** and **age** from facial images in real-time.  
Both models are trained separately on the **[UTKFace Dataset](https://susanqq.github.io/UTKFace/)** and integrated with **OpenCV** for face detection and live camera inference.

---

## Features
- Real-time **face detection** using OpenCV’s Haar Cascade Classifier  
- Two separate CNN models:
  - **Gender Detection Model** — predicts *Male* / *Female*  
  - **Age Detection Model** — predicts the *approximate age* of the person  
- Draws a bounding box around detected faces  
- Displays **predicted gender and age** above the box  

---

## Model Details

| Model | Dataset | Metric | Value |
|:------|:---------|:--------|:-------|
| **Gender Detection** | UTKFace | Accuracy | **91.3%** |
| **Age Detection** | UTKFace | MAE | **6.7** |
|  |  | RMSE | **9.2** |

---

## Dataset — UTKFace
The **UTKFace dataset** is a large-scale face dataset with long age span (0–116 years), containing over 20,000 face images with labels of **age, gender, and ethnicity**.

Each image file name follows this format:  
```
[age]_[gender]_[race]_[date&time].jpg
```

Example:  
```
25_0_1_201701161745.jpg
```
→ Age: 25, Gender: Male (0), Race: White (1)

---

## Tech Stack
- **Python 3.x**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**

---

## How to Run

### Clone this repository
```bash
git clone https://github.com/milan-roy/Pixel-Persona.git
cd Pixel-Persona
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the camera detection script
```bash
python detect_gender_age.py
```

The program will:
- Access your **device camera**
- Detect faces in real-time  
- Display the **predicted gender and age** on the video feed  

Press **‘q’** to quit the application.

---
