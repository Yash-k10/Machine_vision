# 🎯 Optical Flow Motion Analyzer

A real-time desktop application for analyzing motion patterns between consecutive video frames using three optical flow techniques.

---

## ✨ Features

| Feature | Description |
|---------|------------|
| 🎥 **Dual Input** | Webcam (real-time) and video file (mp4/avi/mkv/mov) |
| 🔵 **Lucas-Kanade** | Sparse feature tracking with coloured motion trails |
| 🟢 **Farnebäck** | Dense per-pixel flow with HSV visualisation + arrow overlay |
| 🔴 **Motion Heatmap** | Cumulative heat visualisation with temporal decay |
| 📊 **Live Statistics** | Average/max magnitude, dominant direction, motion area % |
| 📈 **Direction Histogram** | Polar plot showing angular distribution of motion |
| 🎛️ **Parameter Tuning** | Real-time sliders for all algorithm parameters |
| 🌙 **Dark Theme** | Modern Catppuccin Mocha-inspired dark UI |
| 📸 **Screenshot** | Capture processed frame at any time |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+**
- A webcam (optional — you can also load video files)

### Installation

```bash
# Navigate to the project directory
cd TAE1

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

---

## 🎮 Controls

### Menu Bar
- **File → Open Video File** (`Ctrl+O`) — Load a video file
- **File → Open Webcam** (`Ctrl+W`) — Start webcam capture
- **View → Save Screenshot** (`Ctrl+S`) — Save current frame

### Toolbar
- **▶ Play / ⏸ Pause** — Toggle playback
- **⏹ Stop** — Stop and release source
- **📸 Screenshot** — Save current frame
- **🔄 Reset** — Reset all processor state

### Control Panel (Right Side)

**Mode Tab:**
- Switch between Lucas-Kanade, Farnebäck, and Heatmap modes
- Toggle original frame overlay

**Params Tab:**
- Adjust algorithm-specific parameters via sliders
- Changes apply in real-time

**Stats Tab:**
- View motion metrics (magnitude, direction, area)
- Polar direction histogram updates live

---

## 📐 Architecture

```
TAE1/
├── main.py                  # Entry point
├── core/                    # Optical flow engine
│   ├── video_source.py      # Webcam / file abstraction
│   ├── lucas_kanade.py      # Sparse optical flow
│   ├── farneback.py         # Dense optical flow
│   ├── motion_heatmap.py    # Cumulative heatmap
│   └── motion_stats.py      # Motion metrics calculator
├── gui/                     # PyQt5 interface
│   ├── app_window.py        # Main window
│   ├── video_canvas.py      # Frame display widget
│   ├── control_panel.py     # Controls & stats panel
│   └── styles.py            # Dark theme stylesheet
└── utils/
    └── frame_utils.py       # Drawing & conversion helpers
```

---

## 🔬 Optical Flow Techniques

### Lucas-Kanade (Sparse)
Tracks feature points (Shi-Tomasi corners) across frames using pyramidal Lucas-Kanade. Ideal for tracking individual objects or measuring specific point velocities.

### Farnebäck (Dense)
Computes a motion vector for every pixel using polynomial expansion. Visualised as HSV colour coding where **hue** represents direction and **brightness** represents speed.

### Motion Heatmap
Accumulates frame-to-frame differences with exponential temporal decay. Regions with sustained or repeated motion glow brighter. Ideal for surveillance and activity mapping.

---

## 📄 License

This project is for educational purposes (TAE1 — Machine Vision).
