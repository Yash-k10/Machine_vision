# Open Ended Practical: Smart Parking Slot Detection System

Developed by **Yash Kapse** for Machine Vision / Computer Vision practical assessment & viva.

---

## 📌 Project Overview

This project implements an automated **Smart Parking Slot Detection System** using Computer Vision algorithms in Python and OpenCV. 

The system analyzes aerial parking lot feeds, crops each parking slot Region of Interest (ROI), applies adaptive thresholding and edge detection, and counts non-zero pixel density to classify slots as **Occupied 🔴** or **Vacant 🟢**.

---

## ✨ Features

- **Occupancy Classification**: Classifies each bay as **Occupied 🔴** (pixel density > threshold) or **Vacant 🟢** (pixel density ≤ threshold).
- **Real-Time Analytics**: Live display of **Total Slots**, **Available Slots**, **Occupied Slots**, and **Occupancy Rate (%)**.
- **Interactive Web Dashboard (`app.py`)**: Built with FastAPI & modern White/Olive UI featuring `"Made by Yash Kapse"` branding.
- **Synthetic Parking Lot Generator (`samples/generate_parking_lot.py`)**: Generates top-down parking graphics with parked vehicles and empty slots for instant testing and viva demos.
- **Viva-Ready Jupyter Notebook (`open_ended_practical.ipynb`)**: Beginner-friendly, well-commented notebook with step-by-step OpenCV code cells, image processing comparison plots, and viva Q&A.

---

## 📁 Directory Structure

```
Open Ended Practical/
├── core/
│   ├── parking_detector.py   # OpenCV classification engine (Blur, Thresholding, Edge Density)
│   └── slot_picker.py        # Parking slot bounding box coordinate manager & JSON persistence
├── samples/
│   └── generate_parking_lot.py # Generator for synthetic parking lot images
├── open_ended_practical.ipynb # Step-by-step Jupyter Notebook for viva & practical demonstration
├── app.py                    # FastAPI Web Dashboard (Made by Yash Kapse)
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```

---

## 🚀 Quick Start Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Interactive Web Dashboard
```bash
python app.py
```
Open **`http://127.0.0.1:8000`** in your browser.

### 3. Open Jupyter Notebook
```bash
jupyter notebook open_ended_practical.ipynb
```
