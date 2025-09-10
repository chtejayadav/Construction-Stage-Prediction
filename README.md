# 🏗️ Construction Stage Prediction App

This project is a **machine learning web application** that predicts the **stage of a building construction** from an uploaded image. It uses **deep learning (MobileNetV2 + transfer learning)** and a simple **Streamlit interface** to make predictions in real time.

---

## ✨ Features

* 📷 Upload a construction site image (JPG or PNG).
* 🔮 Predicts the stage of construction with confidence score.
* 🧠 Uses **MobileNetV2** as a pretrained backbone for feature extraction.
* 🎨 Clean and user-friendly Streamlit interface with custom styling.
* 🚀 Supports **6 unique construction stages**:

  1. Site Preparation
  2. Foundation Work
  3. Structural Framework
  4. Roofing & Exterior
  5. Interior Work
  6. Final Touches

---

## 🛠️ Tech Stack

* **Python 3.9+**
* **TensorFlow / Keras** → for model training & prediction
* **MobileNetV2** → transfer learning model
* **Streamlit** → interactive web app
* **scikit-learn** → train-test split, preprocessing
* **NumPy** → numerical operations

---

## 📂 Project Structure

```
├── Construction.py          # Main Streamlit app  
├── requirements.txt         # Required dependencies  
├── construction_stage_model.h5  # Saved trained model (auto-generated after training)  
├── dataset/                 # Training images (12 sample images)  
│   ├── 1.jpg … 12.jpg  
└── README.md                # Project documentation  
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/construction-stage-prediction.git
   cd construction-stage-prediction
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**

   ```bash
   streamlit run Construction.py
   ```

4. Open the link provided by Streamlit in your browser (usually `http://localhost:8501`).

---

## 📤 How to Use

1. Launch the app using Streamlit.
2. Upload a clear construction site image (JPG/PNG).
3. The model predicts the **current stage** with a confidence percentage.
4. The app also provides a **stage description** and an **estimated time to completion**.

---

## 📊 Model Details

* Base model: **MobileNetV2** (pretrained on ImageNet).
* Fine-tuned on **12 construction images**.
* Classes: **6 construction stages**.
* Input image size: **128×128 pixels**.

---

## 🚧 Example Prediction Output

✅ **Predicted Stage**: Stage 2 – Structural Framework (Confidence: 0.92)

### Stage 2 – Structural Framework

* Work Done: 33.4% – 50%
* Beams, columns, floors, and slabs are constructed.
* ✅ Work Completed: Load-bearing framework and floors finished.
* ⏳ Estimated Time to Completion: 4.5 to 5 months.

---

## 🙌 Acknowledgments

* TensorFlow/Keras team for MobileNetV2.
* Streamlit community for making ML deployment simple.
* This project is a demo and can be extended with larger datasets for higher accuracy.

