# Practical 9: Government Document Vision & OCR Application

A comprehensive Computer Vision & OCR Application developed for government offices to convert physical scanned documents, identity forms, and certificates into editable digital text.

---

## 📌 Features

1. **Advanced Document Image Preprocessing (`core/preprocessor.py`)**:
   - **Automatic Skew Correction**: Calculates text skew angle using OpenCV minimum area bounding box & Radon line analysis.
   - **Denoising Filters**: Gaussian, Median, and Bilateral noise removal.
   - **Adaptive Thresholding**: Otsu binarization and Sauvola adaptive thresholding.
   - **Perspective Unwarping**: Document quadrangle detection and 4-point perspective transformation.

2. **Multi-Engine OCR & Visual Bounding Boxes (`core/ocr_engine.py`)**:
   - **Pytesseract Integration**: Extract word confidence scores and bounding boxes `[x, y, w, h]`.
   - **EasyOCR Deep Learning Fallback**: Handles rotated or stylized fonts.
   - **Confidence Visualization**: Color-coded bounding box overlays (Green: >80%, Yellow: 50-80%, Red: <50%).

3. **Entity Validation Pipeline (`core/validator.py`)**:
   - **Regex Extraction**: Recognizes Indian Government IDs (Aadhaar 12-digit format, PAN 10-character alphanumeric, Passport), Date formats (`DD/MM/YYYY`), phone numbers, and email addresses.
   - **Key-Value Pair Extraction**: Parses structured fields (Name, Father's Name, DOB, Gender, Issuing Office).
   - **Accuracy Metrics**: Calculates Character Error Rate (CER) and Word Error Rate (WER) using Levenshtein edit distance.

4. **Synthetic Document Generator (`samples/synthetic_docs.py`)**:
   - Generates realistic synthetic government certificates with noise, rotation skew, and seal stamps for testing.

5. **Interactive Web Dashboard (`app.py`)**:
   - Built with **FastAPI** and modern dark-mode glassmorphism UI.
   - Live visual comparison across preprocessing stages.
   - Live editable text area with instant export to **TXT**, **DOCX**, and **JSON**.

---

## 🚀 Installation & Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Interactive Web App
```bash
python app.py
```
Open your browser at `http://127.0.0.1:8000`.

### 3. Run Jupyter Notebook
Open and execute `practical_9.ipynb` for step-by-step visual demonstration and code walkthrough.

---

## 📁 Directory Structure

```
Practical 9/
├── core/
│   ├── preprocessor.py       # Skew correction, denoising, adaptive thresholding
│   ├── ocr_engine.py         # Pytesseract & EasyOCR integration, bbox visualization
│   └── validator.py          # Aadhaar/PAN regex validation, key-value parsing, CER/WER
├── samples/
│   └── synthetic_docs.py     # Generator for scanned government certificates & forms
├── practical_9.ipynb         # Interactive notebook with visual plots and metrics
├── app.py                    # FastAPI web server and modern glassmorphism web dashboard
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```
