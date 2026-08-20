"""
Smart Parking Slot Detection Web Dashboard (FastAPI + White & Olive Theme).

Provides real-time parking slot occupancy monitoring, multi-feature OpenCV classification,
interactive canvas slot box placement, and real-life parking feed analysis.
Made by Yash Kapse.
"""

import os
import sys
import io
import base64
import json
import random
import cv2
import numpy as np
import traceback
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure core modules are importable
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from core.parking_detector import ParkingDetector
from core.slot_picker import SlotManager
from samples.generate_parking_lot import generate_parking_lot_image

app = FastAPI(
    title="Smart Parking Slot Detection System — Made by Yash Kapse",
    description="Detect and classify parking slots as Occupied or Vacant using Computer Vision. Made by Yash Kapse.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

slot_manager = SlotManager()


class ProcessRequest(BaseModel):
    image_base64: Optional[str] = None
    sensitivity: float = 0.5
    auto_detect_slots: bool = False
    custom_slots: Optional[List[Dict[str, Any]]] = None
    blur_kernel: int = 3
    block_size: int = 25
    c_val: int = 16


def mat_to_base64(img: np.ndarray, format: str = ".png") -> str:
    """Helper to convert OpenCV image matrix to base64 string."""
    if img is None:
        return ""
    success, buffer = cv2.imencode(format, img)
    if not success:
        return ""
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.get("/api/generate_sample")
async def generate_sample(occupied_count: int = 6):
    """Generate synthetic parking lot image with specified number of cars."""
    try:
        total = 12
        occupied_count = min(total, max(0, occupied_count))
        occupied_indices = random.sample(range(total), occupied_count)
        
        img, slots, meta = generate_parking_lot_image(rows=2, cols=6, occupied_indices=occupied_indices)
        img_b64 = mat_to_base64(img)

        return {
            "success": True,
            "image_base64": img_b64,
            "slots": slots,
            "metadata": meta
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/api/process")
async def process_parking_frame(req: ProcessRequest):
    """
    Process parking lot image using Multi-Feature OpenCV Classification Engine.
    Supports custom user-defined bounding boxes, auto-contour detection, or resolution-scaled grid layouts.
    """
    try:
        if not req.image_base64:
            return JSONResponse(status_code=400, content={"success": False, "error": "No image provided."})

        clean_b64 = req.image_base64.strip()
        if "," in clean_b64:
            clean_b64 = clean_b64.split(",")[1]

        img_bytes = base64.b64decode(clean_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if raw_img is None:
            return JSONResponse(status_code=400, content={"success": False, "error": "Could not decode parking image."})

        detector = ParkingDetector(sensitivity=req.sensitivity)
        slots = []

        # 1. Custom User-Defined Slots
        if req.custom_slots and len(req.custom_slots) > 0:
            slots = req.custom_slots
        # 2. Auto Line Contour Detection
        elif req.auto_detect_slots:
            slots = detector.detect_automatic_slots(raw_img)

        # 3. Fallback: Resolution-Scaled Multi-Bay Grid
        if not slots or len(slots) == 0:
            base_slots = slot_manager.load_slots()
            img_h, img_w = raw_img.shape[:2]
            base_w, base_h = 780, 530

            scale_x = img_w / base_w
            scale_y = img_h / base_h

            slots = []
            for s in base_slots:
                bx, by, bw, bh = s["bbox"]
                sx = max(0, int(bx * scale_x))
                sy = max(0, int(by * scale_y))
                sw = max(10, min(img_w - sx, int(bw * scale_x)))
                sh = max(10, min(img_h - sy, int(bh * scale_y)))
                slots.append({
                    "id": s["id"],
                    "bbox": [sx, sy, sw, sh]
                })

        res = detector.process_slots(raw_img, slots)

        return {
            "success": True,
            "total_slots": res["total_slots"],
            "occupied_slots": res["occupied_slots"],
            "available_slots": res["available_slots"],
            "occupancy_rate": res["occupancy_rate"],
            "slots": res["slots"],
            "images": {
                "overlay": mat_to_base64(res["overlay_image"]),
                "binary": mat_to_base64(res["binary_image"]),
                "original": mat_to_base64(raw_img)
            }
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": f"Processing Error: {str(e)}"})


@app.post("/api/export")
async def export_report(request: Request):
    """Export parking analytics report as TXT or JSON."""
    try:
        data = await request.json()
        stats = data.get("stats", {})
        fmt = str(data.get("format", "txt")).lower()

        text_lines = [
            "==================================================",
            "   SMART PARKING SLOT DETECTION REPORT",
            "   Made by Yash Kapse — Machine Vision Practical",
            "==================================================",
            f"Total Slots     : {stats.get('total_slots', 0)}",
            f"Occupied Slots 🔴: {stats.get('occupied_slots', 0)}",
            f"Available Slots 🟢: {stats.get('available_slots', 0)}",
            f"Occupancy Rate  : {stats.get('occupancy_rate', 0)}%",
            "--------------------------------------------------",
            "INDIVIDUAL SLOT STATUS BREAKDOWN:",
        ]
        for s in stats.get("slots", []):
            icon = "🔴 OCCUPIED" if s.get("occupied") else "🟢 VACANT"
            text_lines.append(f"Slot {s.get('id'):<2} | Status: {icon:<11} | Confidence: {s.get('confidence')}% | Texture StdDev: {s.get('texture_std_dev')}")

        report_text = "\n".join(text_lines)

        if fmt == "txt":
            stream = io.BytesIO(report_text.encode("utf-8"))
            return StreamingResponse(
                stream,
                media_type="text/plain",
                headers={"Content-Disposition": "attachment; filename=parking_summary.txt"}
            )
        elif fmt == "json":
            json_report = {
                "author": "Made by Yash Kapse",
                "summary": stats
            }
            stream = io.BytesIO(json.dumps(json_report, indent=2).encode("utf-8"))
            return StreamingResponse(
                stream,
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=parking_summary.json"}
            )
        return JSONResponse(status_code=400, content={"error": "Unsupported format."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Render single-page Smart Parking Web UI (White & Olive Theme)."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Parking Detection — Made by Yash Kapse</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
            --panel-bg: #ffffff;
            --panel-border: rgba(85, 107, 47, 0.2);
            --panel-shadow: 0 4px 20px rgba(85, 107, 47, 0.08);
            
            --olive-primary: #556b2f;
            --olive-dark: #3e4e22;
            --olive-light: #6b8e23;
            --olive-soft: rgba(85, 107, 47, 0.08);
            --olive-hover: #485c27;

            --occupied-red: #dc2626;
            --vacant-green: #16a34a;
            
            --text-main: #1e293b;
            --text-muted: #64748b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        header {
            background: #ffffff;
            border-bottom: 2px solid var(--olive-primary);
            padding: 1rem 2.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 10px rgba(85, 107, 47, 0.1);
        }

        .logo-container {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .logo-icon {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, #556b2f, #6b8e23);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            font-size: 22px;
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(85, 107, 47, 0.3);
        }

        .app-title { font-size: 1.35rem; font-weight: 800; color: #1e293b; letter-spacing: -0.5px; }
        .app-subtitle { font-size: 0.82rem; color: var(--text-muted); }

        .author-badge {
            background: linear-gradient(135deg, #556b2f, #6b8e23);
            color: #ffffff;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.3px;
            box-shadow: 0 2px 8px rgba(85, 107, 47, 0.25);
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn {
            background: var(--olive-primary);
            color: white;
            border: none;
            padding: 10px 22px;
            border-radius: 8px;
            font-family: inherit;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 3px 10px rgba(85, 107, 47, 0.25);
        }
        .btn:hover { background: var(--olive-hover); transform: translateY(-1px); box-shadow: 0 5px 15px rgba(85, 107, 47, 0.35); }
        .btn-secondary { background: #f1f5f9; color: var(--olive-dark); border: 1px solid var(--panel-border); box-shadow: none; }
        .btn-secondary:hover { background: #e2e8f0; color: var(--olive-primary); }
        .btn:disabled { opacity: 0.6; cursor: not-allowed; }

        main {
            flex: 1;
            padding: 1.5rem 2.5rem;
            display: grid;
            grid-template-columns: 340px 1fr;
            gap: 1.5rem;
            max-width: 1650px;
            margin: 0 auto;
            width: 100%;
        }

        .sidebar {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 1.4rem;
            display: flex;
            flex-direction: column;
            gap: 1.4rem;
            box-shadow: var(--panel-shadow);
        }

        .section-title {
            font-size: 0.9rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--olive-primary);
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .drop-zone {
            border: 2px dashed var(--olive-primary);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            background: var(--olive-soft);
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .drop-zone:hover { border-color: var(--olive-dark); background: rgba(85, 107, 47, 0.14); }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 12px;
        }
        .control-label { font-size: 0.85rem; font-weight: 600; color: var(--text-main); display: flex; justify-content: space-between; }
        input[type="range"] { width: 100%; accent-color: var(--olive-primary); }

        .toggle-switch {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.85rem;
            font-weight: 600;
            padding: 6px 0;
            color: var(--text-main);
        }

        .content-area {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .metrics-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
        }

        .metric-card {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 4px;
            box-shadow: var(--panel-shadow);
        }
        .metric-val { font-size: 1.7rem; font-weight: 800; color: var(--olive-primary); }
        .metric-val.vacant { color: var(--vacant-green); }
        .metric-val.occupied { color: var(--occupied-red); }
        .metric-lbl { font-size: 0.75rem; color: var(--text-muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }

        .workspace-grid {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 1.5rem;
            flex: 1;
        }

        .panel {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 1.4rem;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            box-shadow: var(--panel-shadow);
        }

        .tabs {
            display: flex;
            gap: 8px;
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 8px;
        }
        .tab {
            padding: 7px 16px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            color: var(--text-muted);
            transition: all 0.2s;
        }
        .tab.active { background: var(--olive-primary); color: white; }

        .img-preview-container {
            flex: 1;
            min-height: 420px;
            background: #f8fafc;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            border: 1px solid #e2e8f0;
            position: relative;
        }
        .img-preview-container img { max-width: 100%; max-height: 540px; object-fit: contain; }

        .slot-grid-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
        }
        .slot-grid-table th { background: #f8fafc; padding: 8px; text-align: left; border-bottom: 2px solid #e2e8f0; color: var(--olive-primary); }
        .slot-grid-table td { padding: 8px; border-bottom: 1px solid #f1f5f9; }

        .badge-status {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 700;
            display: inline-block;
        }
        .badge-vacant { background: rgba(22, 163, 74, 0.12); color: #15803d; border: 1px solid rgba(22, 163, 74, 0.3); }
        .badge-occupied { background: rgba(220, 38, 38, 0.12); color: #b91c1c; border: 1px solid rgba(220, 38, 38, 0.3); }

        footer {
            background: #ffffff;
            border-top: 1px solid var(--panel-border);
            padding: 0.8rem 2.5rem;
            text-align: center;
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 500;
        }
        footer span { color: var(--olive-primary); font-weight: 700; }
    </style>
</head>
<body>
    <header>
        <div class="logo-container">
            <div class="logo-icon">🅿️</div>
            <div>
                <div class="app-title">Smart Parking Slot Detection System</div>
                <div class="app-subtitle">Computer Vision Occupancy Classification & Slot Monitoring</div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 16px;">
            <div class="author-badge">👤 Made by Yash Kapse</div>
            <button class="btn btn-secondary" onclick="generateSampleLot()">🚗 Generate Parking Lot</button>
            <button class="btn" id="btnProcess" onclick="triggerProcess()">🔍 Detect Occupancy</button>
        </div>
    </header>

    <main>
        <div class="sidebar">
            <div>
                <div class="section-title">📷 Parking Feed Input</div>
                <div class="drop-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
                    <span style="font-size: 32px;">🅿️</span>
                    <div style="margin-top: 8px; font-weight: 600; font-size: 0.92rem;" id="fileNameDisplay">Upload Parking Lot Image</div>
                    <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 4px;">PNG, JPG, Aerial Drone/CCTV Stream</div>
                    <input type="file" id="fileInput" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
                </div>
            </div>

            <div>
                <div class="section-title">⚙️ Slot Detection Settings</div>

                <div class="toggle-switch">
                    <span>Auto-Detect Line Outlines</span>
                    <input type="checkbox" id="chkAutoDetectSlots" onchange="triggerProcess()">
                </div>

                <div class="control-group" style="margin-top: 10px;">
                    <div class="control-label">
                        <span>Detection Sensitivity</span>
                        <span id="lblSensitivityVal">0.5</span>
                    </div>
                    <input type="range" id="sensitivityRange" min="0.1" max="1.0" step="0.05" value="0.5" oninput="updateSensitivityLabel(this.value)">
                </div>

                <div style="font-size: 0.78rem; color: var(--text-muted); line-height: 1.4; background: var(--olive-soft); padding: 10px; border-radius: 8px;">
                    💡 <strong>Real-Life Multi-Feature Fusion:</strong> Analyzes inner ROI edge density, Canny contours, and HSV color variance. Works on any real parking photo/video frame.
                </div>
            </div>

            <div>
                <div class="section-title">📊 Analytics Export</div>
                <div style="display: flex; gap: 8px;">
                    <button class="btn btn-secondary" style="flex:1;" onclick="exportAnalytics('txt')">Export TXT</button>
                    <button class="btn btn-secondary" style="flex:1;" onclick="exportAnalytics('json')">Export JSON</button>
                </div>
            </div>
        </div>

        <div class="content-area">
            <div class="metrics-row">
                <div class="metric-card">
                    <div class="metric-lbl">Total Slots</div>
                    <div class="metric-val" id="metricTotal">0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-lbl">Available Bays 🟢</div>
                    <div class="metric-val vacant" id="metricVacant">0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-lbl">Occupied Bays 🔴</div>
                    <div class="metric-val occupied" id="metricOccupied">0</div>
                </div>
                <div class="metric-card">
                    <div class="metric-lbl">Occupancy Rate</div>
                    <div class="metric-val" id="metricRate">0%</div>
                </div>
            </div>

            <div class="workspace-grid">
                <!-- Left Panel: Classified Bounding Box Overlay -->
                <div class="panel">
                    <div class="tabs">
                        <div class="tab active" onclick="switchImageTab('overlay', this)">Classified Overlay (🔴 / 🟢)</div>
                        <div class="tab" onclick="switchImageTab('binary', this)">Edge Binary Map</div>
                        <div class="tab" onclick="switchImageTab('original', this)">Raw Input</div>
                    </div>
                    <div class="img-preview-container">
                        <div id="imagePlaceholder" style="color: var(--text-muted); font-size: 0.9rem;">No parking image loaded</div>
                        <img id="mainImagePreview" style="display: none;" src="" alt="Parking Status">
                    </div>
                </div>

                <!-- Right Panel: Detailed Slot Breakdown Table -->
                <div class="panel">
                    <div class="section-title" style="margin-bottom: 0;">📋 Slot Status Breakdown</div>
                    <div style="flex: 1; overflow-y: auto; max-height: 480px;">
                        <table class="slot-grid-table">
                            <thead>
                                <tr>
                                    <th>Slot ID</th>
                                    <th>Status</th>
                                    <th>Confidence</th>
                                    <th>Texture StdDev</th>
                                </tr>
                            </thead>
                            <tbody id="slotTableBody">
                                <tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No slots analyzed yet.</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <footer>
        Open Ended Practical — Smart Parking Detection | <span>Made by Yash Kapse</span>
    </footer>

    <script>
        let currentData = null;
        let selectedImageBase64 = null;
        let _debounceTimer = null;

        function updateSensitivityLabel(val) {
            document.getElementById('lblSensitivityVal').innerText = val;
            // Debounce: only fire process 400ms after user stops moving slider
            clearTimeout(_debounceTimer);
            _debounceTimer = setTimeout(() => {
                if (selectedImageBase64) triggerProcess();
            }, 400);
        }

        function handleFileSelect(evt) {
            const file = evt.target.files[0];
            if (!file) return;
            document.getElementById('fileNameDisplay').innerText = file.name;

            const reader = new FileReader();
            reader.onload = function(e) {
                selectedImageBase64 = e.target.result;
                document.getElementById('mainImagePreview').src = selectedImageBase64;
                document.getElementById('mainImagePreview').style.display = 'block';
                document.getElementById('imagePlaceholder').style.display = 'none';
                triggerProcess();
            };
            reader.readAsDataURL(file);
        }

        async function generateSampleLot() {
            try {
                document.getElementById('fileNameDisplay').innerText = 'Sample Parking Lot (6 Busy / 6 Free)';
                const res = await fetch('/api/generate_sample?occupied_count=6');
                const data = await res.json();
                if (!data.success) {
                    alert("Failed to generate lot: " + data.error);
                    return;
                }
                selectedImageBase64 = data.image_base64;
                document.getElementById('mainImagePreview').src = selectedImageBase64;
                document.getElementById('mainImagePreview').style.display = 'block';
                document.getElementById('imagePlaceholder').style.display = 'none';
                triggerProcess();
            } catch (err) {
                alert("Generation error: " + err);
            }
        }

        let _processing = false;

        async function triggerProcess() {
            if (!selectedImageBase64 || _processing) return;
            _processing = true;

            const btn = document.getElementById('btnProcess');
            btn.disabled = true;
            btn.innerText = '⏳ Processing...';

            const payload = {
                image_base64: selectedImageBase64,
                sensitivity: parseFloat(document.getElementById('sensitivityRange').value),
                auto_detect_slots: document.getElementById('chkAutoDetectSlots').checked
            };

            try {
                const res = await fetch('/api/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!res.ok) {
                    let errMsg = 'Server error ' + res.status;
                    try { const ej = await res.json(); errMsg = ej.error || errMsg; } catch(e) {}
                    alert('Detection error: ' + errMsg);
                    return;
                }

                const data = await res.json();
                if (!data.success) {
                    alert('Detection error: ' + (data.error || 'Processing failed.'));
                    return;
                }

                currentData = data;
                updateUI(data);
            } catch (err) {
                alert('Connection error: ' + err);
            } finally {
                _processing = false;
                btn.disabled = false;
                btn.innerText = '🔍 Detect Occupancy';
            }
        }

        function updateUI(data) {
            document.getElementById('metricTotal').innerText = data.total_slots;
            document.getElementById('metricVacant').innerText = data.available_slots;
            document.getElementById('metricOccupied').innerText = data.occupied_slots;
            document.getElementById('metricRate').innerText = data.occupancy_rate + '%';

            if (data.images && data.images.overlay) {
                document.getElementById('mainImagePreview').src = data.images.overlay;
            }

            // Populate Table Breakdown
            const tbody = document.getElementById('slotTableBody');
            tbody.innerHTML = '';
            data.slots.forEach(s => {
                const tr = document.createElement('tr');
                const badgeClass = s.occupied ? 'badge-occupied' : 'badge-vacant';
                const statusText = s.occupied ? '🔴 OCCUPIED' : '🟢 VACANT';

                tr.innerHTML = `
                    <td style="font-weight: 700;">Slot ${s.id}</td>
                    <td><span class="badge-status ${badgeClass}">${statusText}</span></td>
                    <td>${s.confidence}%</td>
                    <td>${s.texture_std_dev}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        function switchImageTab(imageKey, el) {
            document.querySelectorAll('.panel:first-child .tab').forEach(t => t.classList.remove('active'));
            el.classList.add('active');
            if (currentData && currentData.images && currentData.images[imageKey]) {
                document.getElementById('mainImagePreview').src = currentData.images[imageKey];
            }
        }

        async function exportAnalytics(format) {
            if (!currentData) {
                alert("No detection data to export.");
                return;
            }
            try {
                const res = await fetch('/api/export', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ stats: currentData, format: format })
                });
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `parking_report.${format}`;
                document.body.appendChild(a);
                a.click();
                a.remove();
            } catch (err) {
                alert("Export error: " + err);
            }
        }

        // Auto load sample lot on page start
        window.addEventListener('load', () => {
            generateSampleLot();
        });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
