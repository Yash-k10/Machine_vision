"""
Control panel widget — right-side panel with mode selection,
parameter sliders, and real-time motion statistics.
"""

import math
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QGroupBox, QRadioButton, QCheckBox, QSlider, QLabel,
    QComboBox, QProgressBar, QGridLayout, QButtonGroup,
    QSizePolicy, QSpacerItem, QFrame,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.motion_stats import MotionMetrics


class ControlPanel(QWidget):
    """Side panel with mode selection, parameter sliders, and statistics."""

    # Signals
    mode_changed = pyqtSignal(str)           # "lucas_kanade" | "farneback" | "heatmap"
    show_original_changed = pyqtSignal(bool)

    # Lucas-Kanade parameter signals
    lk_max_corners_changed = pyqtSignal(int)
    lk_quality_changed = pyqtSignal(float)
    lk_min_distance_changed = pyqtSignal(int)
    lk_win_size_changed = pyqtSignal(int)

    # Farnebäck parameter signals
    fb_winsize_changed = pyqtSignal(int)
    fb_levels_changed = pyqtSignal(int)
    fb_iterations_changed = pyqtSignal(int)
    fb_arrows_changed = pyqtSignal(bool)
    fb_arrow_step_changed = pyqtSignal(int)

    # Heatmap parameter signals
    hm_decay_changed = pyqtSignal(float)
    hm_threshold_changed = pyqtSignal(int)
    hm_colormap_changed = pyqtSignal(str)

    # Object detection signals
    detection_enabled_changed = pyqtSignal(bool)
    det_min_area_changed = pyqtSignal(int)
    det_max_objects_changed = pyqtSignal(int)
    det_show_boxes_changed = pyqtSignal(bool)
    det_show_labels_changed = pyqtSignal(bool)
    det_show_trails_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Title ──
        title = QLabel("⚙  Control Panel")
        title.setStyleSheet(
            "font-size: 16px; font-weight: 700; color: #89b4fa; "
            "padding: 6px 0; background: transparent;"
        )
        layout.addWidget(title)

        # ── Separator ──
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #313244;")
        layout.addWidget(sep)

        # ── Tabs ──
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self._build_mode_tab()
        self._build_params_tab()
        self._build_stats_tab()

    # ==================================================================
    # MODE TAB
    # ==================================================================

    def _build_mode_tab(self):
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.setContentsMargins(10, 14, 10, 10)
        vbox.setSpacing(8)

        # Analysis Mode
        mode_group = QGroupBox("Analysis Mode")
        mode_layout = QVBoxLayout(mode_group)
        mode_layout.setSpacing(10)

        self._mode_group = QButtonGroup(self)
        modes = [
            ("🔵  Lucas-Kanade  (Sparse)", "lucas_kanade"),
            ("🟢  Farnebäck  (Dense)", "farneback"),
            ("🔴  Motion Heatmap", "heatmap"),
        ]
        for i, (label, key) in enumerate(modes):
            rb = QRadioButton(label)
            rb.setProperty("mode_key", key)
            if i == 0:
                rb.setChecked(True)
            self._mode_group.addButton(rb, i)
            mode_layout.addWidget(rb)

        self._mode_group.buttonClicked.connect(self._on_mode_changed)
        vbox.addWidget(mode_group)

        # Show original toggle
        self._show_original_cb = QCheckBox("Show original frame overlay")
        self._show_original_cb.toggled.connect(self.show_original_changed.emit)
        vbox.addWidget(self._show_original_cb)

        # Object detection toggle
        det_group = QGroupBox("Object Detection")
        det_layout = QVBoxLayout(det_group)
        det_layout.setSpacing(6)

        self._detect_cb = QCheckBox("🎯  Enable Object Detection")
        self._detect_cb.setChecked(True)
        self._detect_cb.setStyleSheet("font-weight: 600; font-size: 13px;")
        self._detect_cb.toggled.connect(self.detection_enabled_changed.emit)
        det_layout.addWidget(self._detect_cb)

        det_desc = QLabel(
            "Detects moving objects with bounding boxes, IDs, "
            "speed, and direction arrows."
        )
        det_desc.setWordWrap(True)
        det_desc.setStyleSheet(
            "color: #a6adc8; font-size: 11px; padding: 4px; background: transparent;"
        )
        det_layout.addWidget(det_desc)
        vbox.addWidget(det_group)

        # Mode description
        self._mode_desc = QLabel()
        self._mode_desc.setWordWrap(True)
        self._mode_desc.setStyleSheet(
            "color: #a6adc8; font-size: 11px; padding: 8px; "
            "background-color: #181825; border-radius: 6px;"
        )
        self._update_mode_desc("lucas_kanade")
        vbox.addWidget(self._mode_desc)

        vbox.addStretch()
        self._tabs.addTab(tab, "Mode")

    def _on_mode_changed(self, button):
        key = button.property("mode_key")
        self._update_mode_desc(key)
        self._update_param_visibility(key)
        self.mode_changed.emit(key)

    def _update_mode_desc(self, key):
        descs = {
            "lucas_kanade": (
                "Tracks sparse feature points (Shi-Tomasi corners) across "
                "frames using pyramidal Lucas-Kanade. Best for tracking "
                "individual objects or key points."
            ),
            "farneback": (
                "Computes a dense per-pixel motion field. Visualised as "
                "HSV colour (hue = direction, brightness = speed). "
                "Best for understanding full-frame motion patterns."
            ),
            "heatmap": (
                "Accumulates motion over time with temporal decay. "
                "Hot regions indicate sustained or repeated motion. "
                "Best for surveillance and activity analysis."
            ),
        }
        self._mode_desc.setText(descs.get(key, ""))

    # ==================================================================
    # PARAMETERS TAB
    # ==================================================================

    def _build_params_tab(self):
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.setContentsMargins(10, 14, 10, 10)
        vbox.setSpacing(10)

        # ── Lucas-Kanade params ──
        self._lk_group = QGroupBox("Lucas-Kanade Parameters")
        lk_layout = QGridLayout(self._lk_group)
        lk_layout.setSpacing(6)

        self._lk_corners_slider = self._add_slider(
            lk_layout, 0, "Max Corners", 10, 500, 200, self._on_lk_corners
        )
        self._lk_quality_slider = self._add_slider(
            lk_layout, 1, "Quality Level", 1, 100, 30, self._on_lk_quality
        )
        self._lk_mindist_slider = self._add_slider(
            lk_layout, 2, "Min Distance", 1, 50, 7, self._on_lk_mindist
        )
        self._lk_winsize_slider = self._add_slider(
            lk_layout, 3, "Window Size", 5, 51, 15, self._on_lk_winsize
        )
        vbox.addWidget(self._lk_group)

        # ── Farnebäck params ──
        self._fb_group = QGroupBox("Farnebäck Parameters")
        fb_layout = QGridLayout(self._fb_group)
        fb_layout.setSpacing(6)

        self._fb_winsize_slider = self._add_slider(
            fb_layout, 0, "Window Size", 5, 51, 15, self._on_fb_winsize
        )
        self._fb_levels_slider = self._add_slider(
            fb_layout, 1, "Pyramid Levels", 1, 10, 3, self._on_fb_levels
        )
        self._fb_iter_slider = self._add_slider(
            fb_layout, 2, "Iterations", 1, 20, 3, self._on_fb_iterations
        )
        self._fb_arrow_step_slider = self._add_slider(
            fb_layout, 3, "Arrow Spacing", 4, 40, 16, self._on_fb_arrow_step
        )
        self._fb_arrows_cb = QCheckBox("Show Arrows")
        self._fb_arrows_cb.setChecked(True)
        self._fb_arrows_cb.toggled.connect(self.fb_arrows_changed.emit)
        fb_layout.addWidget(self._fb_arrows_cb, 4, 0, 1, 3)
        vbox.addWidget(self._fb_group)
        self._fb_group.hide()

        # ── Heatmap params ──
        self._hm_group = QGroupBox("Heatmap Parameters")
        hm_layout = QGridLayout(self._hm_group)
        hm_layout.setSpacing(6)

        self._hm_decay_slider = self._add_slider(
            hm_layout, 0, "Decay Factor", 50, 100, 95, self._on_hm_decay
        )
        self._hm_threshold_slider = self._add_slider(
            hm_layout, 1, "Threshold", 0, 100, 15, self._on_hm_threshold
        )

        # Colormap selector
        hm_layout.addWidget(QLabel("Colormap:"), 2, 0)
        self._hm_cmap_combo = QComboBox()
        self._hm_cmap_combo.addItems(["INFERNO", "JET", "HOT", "MAGMA", "TURBO", "VIRIDIS"])
        self._hm_cmap_combo.currentTextChanged.connect(self.hm_colormap_changed.emit)
        hm_layout.addWidget(self._hm_cmap_combo, 2, 1, 1, 2)
        vbox.addWidget(self._hm_group)
        self._hm_group.hide()

        # ── Object Detection params ──
        self._det_group = QGroupBox("Object Detection Parameters")
        det_p_layout = QGridLayout(self._det_group)
        det_p_layout.setSpacing(6)

        self._det_min_area_slider = self._add_slider(
            det_p_layout, 0, "Min Area", 100, 5000, 800, self._on_det_min_area
        )
        self._det_max_obj_slider = self._add_slider(
            det_p_layout, 1, "Max Objects", 1, 50, 20, self._on_det_max_objects
        )

        self._det_boxes_cb = QCheckBox("Show Bounding Boxes")
        self._det_boxes_cb.setChecked(True)
        self._det_boxes_cb.toggled.connect(self.det_show_boxes_changed.emit)
        det_p_layout.addWidget(self._det_boxes_cb, 2, 0, 1, 3)

        self._det_labels_cb = QCheckBox("Show ID & Speed Labels")
        self._det_labels_cb.setChecked(True)
        self._det_labels_cb.toggled.connect(self.det_show_labels_changed.emit)
        det_p_layout.addWidget(self._det_labels_cb, 3, 0, 1, 3)

        self._det_trails_cb = QCheckBox("Show Motion Trails")
        self._det_trails_cb.setChecked(True)
        self._det_trails_cb.toggled.connect(self.det_show_trails_changed.emit)
        det_p_layout.addWidget(self._det_trails_cb, 4, 0, 1, 3)

        vbox.addWidget(self._det_group)

        vbox.addStretch()
        self._tabs.addTab(tab, "Params")

    def _update_param_visibility(self, mode_key):
        self._lk_group.setVisible(mode_key == "lucas_kanade")
        self._fb_group.setVisible(mode_key == "farneback")
        self._hm_group.setVisible(mode_key == "heatmap")

    # ── Slider factory ──

    def _add_slider(self, layout, row, label_text, min_val, max_val, default, callback):
        label = QLabel(label_text)
        label.setStyleSheet("font-size: 12px;")
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)

        val_label = QLabel(str(default))
        val_label.setFixedWidth(36)
        val_label.setAlignment(Qt.AlignCenter)
        val_label.setStyleSheet("color: #a6e3a1; font-weight: 600; font-size: 12px;")

        def on_change(v):
            val_label.setText(str(v))
            callback(v)

        slider.valueChanged.connect(on_change)

        layout.addWidget(label, row, 0)
        layout.addWidget(slider, row, 1)
        layout.addWidget(val_label, row, 2)
        return slider

    # ── Parameter callbacks ──

    def _on_lk_corners(self, v):
        self.lk_max_corners_changed.emit(v)

    def _on_lk_quality(self, v):
        self.lk_quality_changed.emit(v / 100.0)

    def _on_lk_mindist(self, v):
        self.lk_min_distance_changed.emit(v)

    def _on_lk_winsize(self, v):
        self.lk_win_size_changed.emit(v | 1)  # ensure odd

    def _on_fb_winsize(self, v):
        self.fb_winsize_changed.emit(v | 1)

    def _on_fb_levels(self, v):
        self.fb_levels_changed.emit(v)

    def _on_fb_iterations(self, v):
        self.fb_iterations_changed.emit(v)

    def _on_fb_arrow_step(self, v):
        self.fb_arrow_step_changed.emit(v)

    def _on_hm_decay(self, v):
        self.hm_decay_changed.emit(v / 100.0)

    def _on_hm_threshold(self, v):
        self.hm_threshold_changed.emit(v)

    def _on_det_min_area(self, v):
        self.det_min_area_changed.emit(v)

    def _on_det_max_objects(self, v):
        self.det_max_objects_changed.emit(v)

    # ==================================================================
    # STATISTICS TAB
    # ==================================================================

    def _build_stats_tab(self):
        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.setContentsMargins(10, 14, 10, 10)
        vbox.setSpacing(10)

        # Stats grid
        stats_group = QGroupBox("Motion Metrics")
        grid = QGridLayout(stats_group)
        grid.setSpacing(8)

        self._stat_labels = {}
        stats_def = [
            ("avg_mag",   "Avg Magnitude",  "0.00 px"),
            ("max_mag",   "Max Magnitude",  "0.00 px"),
            ("direction", "Dominant Dir",   "0°"),
            ("points",    "Tracked Points", "0"),
        ]
        for i, (key, name, default) in enumerate(stats_def):
            name_lbl = QLabel(name)
            name_lbl.setObjectName("stat_label")
            val_lbl = QLabel(default)
            val_lbl.setObjectName("stat_value")
            grid.addWidget(name_lbl, i, 0)
            grid.addWidget(val_lbl, i, 1)
            self._stat_labels[key] = val_lbl

        vbox.addWidget(stats_group)

        # Motion area progress bar
        area_group = QGroupBox("Motion Area")
        area_layout = QVBoxLayout(area_group)
        self._motion_area_bar = QProgressBar()
        self._motion_area_bar.setRange(0, 100)
        self._motion_area_bar.setValue(0)
        self._motion_area_bar.setFormat("%v%")
        area_layout.addWidget(self._motion_area_bar)
        vbox.addWidget(area_group)

        # Direction histogram (polar plot)
        hist_group = QGroupBox("Direction Distribution")
        hist_layout = QVBoxLayout(hist_group)

        self._hist_figure = Figure(figsize=(2.8, 2.8), dpi=80)
        self._hist_figure.patch.set_facecolor("#1e1e2e")
        self._hist_canvas = FigureCanvas(self._hist_figure)
        self._hist_canvas.setFixedHeight(220)
        hist_layout.addWidget(self._hist_canvas)
        vbox.addWidget(hist_group)

        self._init_polar_plot()

        vbox.addStretch()
        self._tabs.addTab(tab, "Stats")

    def _init_polar_plot(self):
        """Initialise the polar histogram plot."""
        self._hist_figure.clear()
        self._polar_ax = self._hist_figure.add_subplot(111, projection="polar")
        self._polar_ax.set_facecolor("#181825")
        self._polar_ax.tick_params(colors="#585b70", labelsize=8)
        self._polar_ax.spines["polar"].set_color("#45475a")
        self._polar_ax.set_yticklabels([])

        # Direction labels
        directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        self._polar_ax.set_xticks(angles)
        self._polar_ax.set_xticklabels(directions, color="#a6adc8", fontsize=9)

        # Initial empty bars
        self._polar_bars = self._polar_ax.bar(
            angles, [0] * 8,
            width=2 * np.pi / 8 * 0.85,
            bottom=0,
            color="#89b4fa",
            alpha=0.75,
            edgecolor="#b4d0fb",
            linewidth=0.5,
        )
        self._hist_canvas.draw()

    def update_stats(self, metrics: MotionMetrics):
        """Update all statistics displays from a MotionMetrics object."""
        self._stat_labels["avg_mag"].setText(f"{metrics.avg_magnitude:.2f} px")
        self._stat_labels["max_mag"].setText(f"{metrics.max_magnitude:.2f} px")
        self._stat_labels["direction"].setText(f"{metrics.dominant_direction_deg:.0f}°")
        self._stat_labels["points"].setText(str(metrics.total_points))

        self._motion_area_bar.setValue(int(min(100, metrics.motion_area_pct)))

        # Update polar histogram
        hist = metrics.direction_histogram
        if len(hist) == 8:
            for bar, val in zip(self._polar_bars, hist):
                bar.set_height(val)

            max_val = max(hist) if max(hist) > 0 else 1.0
            self._polar_ax.set_ylim(0, max_val * 1.2)

            # Color-code bars by magnitude
            cmap = [
                "#89b4fa", "#89dceb", "#a6e3a1", "#f9e2af",
                "#fab387", "#f38ba8", "#cba6f7", "#f5c2e7",
            ]
            for bar, val, color in zip(self._polar_bars, hist, cmap):
                bar.set_color(color)
                bar.set_alpha(0.5 + 0.5 * (val / max_val) if max_val > 0 else 0.5)

            self._hist_canvas.draw_idle()
