"""
Main application window — orchestrates video source, optical flow
processors, and the GUI.
"""

import os
import time
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QSplitter,
    QAction, QFileDialog, QStatusBar, QToolBar, QToolButton,
    QMessageBox, QApplication,
)

from gui.video_canvas import VideoCanvas
from gui.control_panel import ControlPanel
from gui.styles import DARK_STYLESHEET

from core.video_source import VideoSource
from core.lucas_kanade import LucasKanadeTracker
from core.farneback import FarnebackAnalyzer
from core.motion_heatmap import MotionHeatmap
from core.motion_stats import MotionStatsCalculator, MotionMetrics
from core.motion_detector import MotionDetector

from utils.frame_utils import draw_info_box


class AppWindow(QMainWindow):
    """Main application window for Optical Flow Motion Analyzer."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Optical Flow Motion Analyzer")
        self.setMinimumSize(1100, 700)
        self.resize(1280, 780)

        # ── Core objects ──
        self._source = VideoSource()
        self._lk_tracker = LucasKanadeTracker()
        self._fb_analyzer = FarnebackAnalyzer()
        self._heatmap = MotionHeatmap()
        self._detector = MotionDetector()
        self._current_mode = "lucas_kanade"
        self._show_original = False
        self._detection_enabled = True

        # FPS tracking
        self._fps_counter = 0
        self._fps_timer = time.time()
        self._current_fps = 0.0

        # Playback state
        self._playing = False

        # ── Build UI ──
        self._build_menu()
        self._build_toolbar()
        self._build_central()
        self._build_statusbar()

        # Apply stylesheet
        self.setStyleSheet(DARK_STYLESHEET)

        # ── Frame timer ──
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._process_frame)

        # ── Connect control panel signals ──
        self._connect_signals()

    # ==================================================================
    # UI CONSTRUCTION
    # ==================================================================

    def _build_menu(self):
        menu = self.menuBar()

        # File menu
        file_menu = menu.addMenu("&File")

        open_video = QAction("📂  Open Video File…", self)
        open_video.setShortcut("Ctrl+O")
        open_video.triggered.connect(self._open_video_file)
        file_menu.addAction(open_video)

        open_webcam = QAction("📷  Open Webcam", self)
        open_webcam.setShortcut("Ctrl+W")
        open_webcam.triggered.connect(self._open_webcam)
        file_menu.addAction(open_webcam)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menu.addMenu("&View")
        screenshot = QAction("📸  Save Screenshot", self)
        screenshot.setShortcut("Ctrl+S")
        screenshot.triggered.connect(self._save_screenshot)
        view_menu.addAction(screenshot)

    def _build_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self._btn_play = QToolButton()
        self._btn_play.setText("▶ Play")
        self._btn_play.setCheckable(True)
        self._btn_play.clicked.connect(self._toggle_playback)
        toolbar.addWidget(self._btn_play)

        self._btn_stop = QToolButton()
        self._btn_stop.setText("⏹ Stop")
        self._btn_stop.clicked.connect(self._stop_source)
        toolbar.addWidget(self._btn_stop)

        toolbar.addSeparator()

        btn_screenshot = QToolButton()
        btn_screenshot.setText("📸 Screenshot")
        btn_screenshot.clicked.connect(self._save_screenshot)
        toolbar.addWidget(btn_screenshot)

        toolbar.addSeparator()

        self._btn_reset = QToolButton()
        self._btn_reset.setText("🔄 Reset")
        self._btn_reset.clicked.connect(self._reset_processors)
        toolbar.addWidget(self._btn_reset)

    def _build_central(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)

        # Video canvas (left, expanding)
        self._canvas = VideoCanvas()
        splitter.addWidget(self._canvas)

        # Control panel (right, fixed)
        self._panel = ControlPanel()
        splitter.addWidget(self._panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        layout.addWidget(splitter)

    def _build_statusbar(self):
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)

        self._status_fps = "FPS: --"
        self._status_frame = "Frame: --"
        self._status_res = "-- × --"
        self._status_source = "No source"

        self._update_statusbar()

    def _update_statusbar(self):
        self._statusbar.showMessage(
            f"  {self._status_source}  │  {self._status_res}  │  "
            f"{self._status_frame}  │  {self._status_fps}"
        )

    # ==================================================================
    # SIGNAL CONNECTIONS
    # ==================================================================

    def _connect_signals(self):
        p = self._panel

        # Mode
        p.mode_changed.connect(self._on_mode_changed)
        p.show_original_changed.connect(self._on_show_original)

        # LK params
        p.lk_max_corners_changed.connect(self._lk_tracker.set_max_corners)
        p.lk_quality_changed.connect(self._lk_tracker.set_quality_level)
        p.lk_min_distance_changed.connect(self._lk_tracker.set_min_distance)
        p.lk_win_size_changed.connect(self._lk_tracker.set_win_size)

        # Farnebäck params
        p.fb_winsize_changed.connect(self._fb_analyzer.set_winsize)
        p.fb_levels_changed.connect(self._fb_analyzer.set_levels)
        p.fb_iterations_changed.connect(self._fb_analyzer.set_iterations)
        p.fb_arrows_changed.connect(lambda v: setattr(self._fb_analyzer, 'show_arrows', v))
        p.fb_arrow_step_changed.connect(self._fb_analyzer.set_arrow_step)

        # Heatmap params
        p.hm_decay_changed.connect(self._heatmap.set_decay)
        p.hm_threshold_changed.connect(self._heatmap.set_threshold)
        p.hm_colormap_changed.connect(self._heatmap.set_colormap)

        # Object detection params
        p.detection_enabled_changed.connect(self._on_detection_toggled)
        p.det_min_area_changed.connect(self._detector.set_min_area)
        p.det_max_objects_changed.connect(self._detector.set_max_objects)
        p.det_show_boxes_changed.connect(lambda v: setattr(self._detector, 'show_boxes', v))
        p.det_show_labels_changed.connect(lambda v: setattr(self._detector, 'show_labels', v))
        p.det_show_trails_changed.connect(lambda v: setattr(self._detector, 'show_trails', v))

    # ==================================================================
    # SOURCE MANAGEMENT
    # ==================================================================

    def _open_video_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv *.flv);;All Files (*)"
        )
        if path:
            self._stop_source()
            if self._source.open_file(path):
                self._status_source = os.path.basename(path)
                self._status_res = f"{self._source.width} × {self._source.height}"
                self._reset_processors()
                self._start_playback()
            else:
                QMessageBox.warning(self, "Error", f"Cannot open video file:\n{path}")

    def _open_webcam(self):
        self._stop_source()
        if self._source.open_webcam(0):
            self._status_source = "Webcam 0"
            self._status_res = f"{self._source.width} × {self._source.height}"
            self._reset_processors()
            self._start_playback()
        else:
            QMessageBox.warning(
                self, "Error",
                "Cannot open webcam.\nMake sure a camera is connected."
            )

    def _stop_source(self):
        self._timer.stop()
        self._playing = False
        self._btn_play.setChecked(False)
        self._btn_play.setText("▶ Play")
        self._source.release()
        self._status_source = "No source"
        self._update_statusbar()

    def _start_playback(self):
        fps = self._source.fps
        interval = max(1, int(1000 / fps))
        self._timer.start(interval)
        self._playing = True
        self._btn_play.setChecked(True)
        self._btn_play.setText("⏸ Pause")
        self._update_statusbar()

    def _toggle_playback(self):
        if not self._source.is_open:
            return
        if self._playing:
            self._timer.stop()
            self._playing = False
            self._btn_play.setText("▶ Play")
            self._btn_play.setChecked(False)
        else:
            self._start_playback()

    # ==================================================================
    # FRAME PROCESSING LOOP
    # ==================================================================

    def _process_frame(self):
        ret, frame = self._source.read_frame()
        if not ret:
            if not self._source.is_webcam:
                # End of video — loop back
                self._source.seek_to(0)
                self._reset_processors()
                return
            return

        # Process based on current mode
        metrics = MotionMetrics()

        if self._current_mode == "lucas_kanade":
            vis, vectors = self._lk_tracker.process(frame)
            if vectors:
                metrics = MotionStatsCalculator.from_sparse_vectors(vectors)

        elif self._current_mode == "farneback":
            vis, flow = self._fb_analyzer.process(frame)
            if flow is not None:
                metrics = MotionStatsCalculator.from_dense_flow(flow)

        elif self._current_mode == "heatmap":
            vis, magnitude = self._heatmap.process(frame)
            if magnitude is not None:
                # Create a pseudo-flow for stats from magnitude
                # Use frame difference magnitude only
                metrics = MotionMetrics(
                    avg_magnitude=float(np.mean(magnitude[magnitude > 0])) if np.any(magnitude > 0) else 0,
                    max_magnitude=float(np.max(magnitude)),
                    motion_area_pct=float(np.sum(magnitude > 5) / magnitude.size * 100),
                    total_points=int(np.sum(magnitude > 5)),
                )
        else:
            vis = frame.copy()

        # Show original overlay if requested
        if self._show_original and self._current_mode != "heatmap":
            vis = cv2.addWeighted(frame, 0.35, vis, 0.65, 0)

        # Object detection overlay
        detected_count = 0
        if self._detection_enabled:
            vis, detected_objects = self._detector.detect(vis)
            detected_count = len(detected_objects)

        # Draw HUD info
        mode_names = {
            "lucas_kanade": "Lucas-Kanade (Sparse)",
            "farneback": "Farneback (Dense)",
            "heatmap": "Motion Heatmap",
        }
        info_lines = [
            f"Mode: {mode_names.get(self._current_mode, '')}",
            f"FPS: {self._current_fps:.1f}",
        ]
        if self._detection_enabled:
            info_lines.append(f"Objects: {detected_count}")
        if not self._source.is_webcam:
            total = self._source.frame_count
            current = self._source.current_frame
            info_lines.append(f"Frame: {current}/{total}")

        draw_info_box(vis, info_lines, "top-left")

        # Display
        self._canvas.update_frame(vis)

        # Update stats panel
        self._panel.update_stats(metrics)

        # FPS calculation
        self._fps_counter += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 0.5:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_timer = time.time()

        # Update status bar
        self._status_fps = f"FPS: {self._current_fps:.1f}"
        if not self._source.is_webcam:
            self._status_frame = f"Frame: {self._source.current_frame}/{self._source.frame_count}"
        else:
            self._status_frame = "Live"
        self._update_statusbar()

    # ==================================================================
    # ACTIONS
    # ==================================================================

    def _on_mode_changed(self, mode_key):
        self._current_mode = mode_key
        self._reset_processors()

    def _on_show_original(self, checked):
        self._show_original = checked

    def _on_detection_toggled(self, enabled):
        self._detection_enabled = enabled
        if not enabled:
            self._detector.reset()

    def _reset_processors(self):
        self._lk_tracker.reset()
        self._fb_analyzer.reset()
        self._heatmap.reset()
        self._detector.reset()

    def _save_screenshot(self):
        pixmap = self._canvas.pixmap()
        if pixmap is None:
            QMessageBox.information(self, "Screenshot", "No frame to capture.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Screenshot", "screenshot.png",
            "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)"
        )
        if path:
            pixmap.save(path)
            self._statusbar.showMessage(f"Screenshot saved: {path}", 3000)

    # ==================================================================
    # CLEANUP
    # ==================================================================

    def closeEvent(self, event):
        self._timer.stop()
        self._source.release()
        event.accept()
