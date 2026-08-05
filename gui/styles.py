"""
Dark theme QSS stylesheet — Catppuccin Mocha inspired.
"""

DARK_STYLESHEET = """
/* ===== Global ===== */
QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}

QMainWindow {
    background-color: #1e1e2e;
}

/* ===== Menu Bar ===== */
QMenuBar {
    background-color: #181825;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
    padding: 2px;
}

QMenuBar::item {
    padding: 5px 12px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #45475a;
}

QMenu {
    background-color: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px;
}

QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}

QMenu::item:selected {
    background-color: #89b4fa;
    color: #1e1e2e;
}

/* ===== Tool Bar ===== */
QToolBar {
    background-color: #181825;
    border-bottom: 1px solid #313244;
    spacing: 6px;
    padding: 4px 8px;
}

QToolButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}

QToolButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
}

QToolButton:pressed {
    background-color: #89b4fa;
    color: #1e1e2e;
}

QToolButton:checked {
    background-color: #a6e3a1;
    color: #1e1e2e;
    border-color: #a6e3a1;
}

/* ===== Status Bar ===== */
QStatusBar {
    background-color: #181825;
    color: #a6adc8;
    border-top: 1px solid #313244;
    padding: 2px;
    font-size: 12px;
}

/* ===== Group Box ===== */
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: 600;
    font-size: 13px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #89b4fa;
}

/* ===== Tabs ===== */
QTabWidget::pane {
    border: 1px solid #45475a;
    border-radius: 6px;
    background-color: #1e1e2e;
}

QTabBar::tab {
    background-color: #181825;
    color: #a6adc8;
    border: 1px solid #313244;
    border-bottom: none;
    padding: 8px 16px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-weight: 500;
}

QTabBar::tab:selected {
    background-color: #1e1e2e;
    color: #89b4fa;
    border-color: #45475a;
    border-bottom: 2px solid #89b4fa;
}

QTabBar::tab:hover:!selected {
    background-color: #313244;
    color: #cdd6f4;
}

/* ===== Radio Buttons ===== */
QRadioButton {
    spacing: 8px;
    padding: 4px;
    font-size: 13px;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #585b70;
    border-radius: 9px;
    background-color: #313244;
}

QRadioButton::indicator:checked {
    background-color: #89b4fa;
    border-color: #89b4fa;
}

QRadioButton::indicator:hover {
    border-color: #89b4fa;
}

/* ===== Check Boxes ===== */
QCheckBox {
    spacing: 8px;
    padding: 4px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #585b70;
    border-radius: 4px;
    background-color: #313244;
}

QCheckBox::indicator:checked {
    background-color: #a6e3a1;
    border-color: #a6e3a1;
}

/* ===== Sliders ===== */
QSlider::groove:horizontal {
    height: 6px;
    background-color: #313244;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -5px 0;
    background-color: #89b4fa;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background-color: #b4d0fb;
}

QSlider::sub-page:horizontal {
    background-color: #89b4fa;
    border-radius: 3px;
}

/* ===== Combo Box ===== */
QComboBox {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px 10px;
    min-width: 80px;
}

QComboBox:hover {
    border-color: #89b4fa;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #1e1e2e;
    border: 1px solid #45475a;
    border-radius: 4px;
    selection-background-color: #89b4fa;
    selection-color: #1e1e2e;
}

/* ===== Labels ===== */
QLabel {
    color: #cdd6f4;
    background-color: transparent;
}

QLabel#stat_value {
    color: #a6e3a1;
    font-weight: 600;
    font-size: 15px;
}

QLabel#stat_label {
    color: #a6adc8;
    font-size: 11px;
}

QLabel#mode_label {
    color: #f9e2af;
    font-weight: 700;
    font-size: 14px;
}

/* ===== Scroll Area ===== */
QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #181825;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background-color: #45475a;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #585b70;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* ===== Push Button ===== */
QPushButton {
    background-color: #313244;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 7px 18px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #45475a;
    border-color: #89b4fa;
}

QPushButton:pressed {
    background-color: #89b4fa;
    color: #1e1e2e;
}

QPushButton#accent_btn {
    background-color: #89b4fa;
    color: #1e1e2e;
    border-color: #89b4fa;
    font-weight: 600;
}

QPushButton#accent_btn:hover {
    background-color: #b4d0fb;
}

QPushButton#danger_btn {
    background-color: #f38ba8;
    color: #1e1e2e;
    border-color: #f38ba8;
}

/* ===== Progress Bar ===== */
QProgressBar {
    background-color: #313244;
    border-radius: 4px;
    text-align: center;
    color: #cdd6f4;
    height: 14px;
    font-size: 11px;
}

QProgressBar::chunk {
    background-color: #a6e3a1;
    border-radius: 4px;
}

/* ===== Splitter ===== */
QSplitter::handle {
    background-color: #313244;
    width: 3px;
}

QSplitter::handle:hover {
    background-color: #89b4fa;
}
"""
