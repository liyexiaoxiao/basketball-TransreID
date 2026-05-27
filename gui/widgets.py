import os
from PyQt5.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPlainTextEdit, 
    QListWidget, QListWidgetItem, QLineEdit, QSplitter
)
from PyQt5.QtGui import QPixmap, QIcon, QColor, QFont
from PyQt5.QtCore import QSize, Qt, pyqtSignal

class ImageCard(QFrame):
    """
    A premium card displaying an image, its ID, Camera, and similarity score.
    Has custom styling and hover effects.
    """
    def __init__(self, image_path, pid, camid, score, is_same_id=None, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.pid = pid
        self.camid = camid
        self.score = score
        self.is_same_id = is_same_id
        
        self.init_ui()

    def init_ui(self):
        self.setFixedWidth(160)
        self.setFixedHeight(230)
        self.setObjectName("ImageCard")
        
        # Border color based on match status
        border_color = "#3d3d3d"  # Default gray
        if self.is_same_id is True:
            border_color = "#2ecc71"  # Green for correct match
        elif self.is_same_id is False:
            border_color = "#e74c3c"  # Red for incorrect match
            
        self.setStyleSheet(f"""
            QFrame#ImageCard {{
                border: 2px solid {border_color};
                border-radius: 8px;
                background-color: #1e1e1e;
            }}
            QFrame#ImageCard:hover {{
                border: 2px solid #3498db;
                background-color: #2a2a2a;
            }}
            QLabel {{
                color: #e0e0e0;
                border: none;
                background-color: transparent;
            }}
            QLabel#ScoreLabel {{
                color: #f1c40f;
                font-weight: bold;
            }}
            QLabel#PidLabel {{
                font-weight: bold;
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        
        # Image Display
        self.img_label = QLabel(self)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("border-radius: 4px; background-color: #121212; border: 1px solid #2a2a2a;")
        
        # Load image asynchronously or synchronously
        if os.path.exists(self.image_path):
            pixmap = QPixmap(self.image_path)
            # Scale maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                144, 150, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.img_label.setPixmap(scaled_pixmap)
        else:
            self.img_label.setText("No Image")
            self.img_label.setFont(QFont("Segoe UI", 9))
            
        layout.addWidget(self.img_label, 1)
        
        # ID and Cam info
        info_layout = QHBoxLayout()
        info_layout.setSpacing(0)
        info_layout.setContentsMargins(0, 0, 0, 0)
        
        pid_text = f"ID: {self.pid}" if self.pid != -1 else "ID: Unknown"
        self.pid_label = QLabel(pid_text, self)
        self.pid_label.setObjectName("PidLabel")
        self.pid_label.setFont(QFont("Segoe UI", 9))
        info_layout.addWidget(self.pid_label)
        
        cam_text = f"Cam: {self.camid}" if self.camid != -1 else "Cam: -"
        self.cam_label = QLabel(cam_text, self)
        self.cam_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.cam_label.setFont(QFont("Segoe UI", 9))
        self.cam_label.setStyleSheet("color: #888888;")
        info_layout.addWidget(self.cam_label)
        
        layout.addLayout(info_layout)
        
        # Distance / Similarity Score
        score_text = f"Sim: {self.score * 100:.1f}%"
        self.score_label = QLabel(score_text, self)
        self.score_label.setObjectName("ScoreLabel")
        self.score_label.setFont(QFont("Segoe UI", 9))
        self.score_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.score_label)

class LogViewer(QPlainTextEdit):
    """
    A dark theme, terminal-style console logging display.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.setStyleSheet("""
            QPlainTextEdit {
                background-color: #0c0c0c;
                color: #00ff00;
                border: 1px solid #333333;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        self.append_log("System initialized.")

    def append_log(self, text):
        from datetime import datetime
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        self.appendPlainText(f"{timestamp} {text}")
        # Scroll to bottom
        self.ensureCursorVisible()

class DatasetListWidget(QFrame):
    """
    Browse a directory of ReID images with thumbnails and search filter.
    """
    image_selected = pyqtSignal(str)

    def __init__(self, directory_path, parent=None):
        super().__init__(parent)
        self.directory_path = directory_path
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet("""
            QFrame {
                background-color: #121212;
                border: none;
            }
            QLineEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 6px 10px;
                font-family: 'Segoe UI';
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #3498db;
            }
            QListWidget {
                background-color: #1a1a1a;
                border: 1px solid #2d2d2d;
                border-radius: 6px;
                padding: 5px;
            }
            QListWidget::item {
                background-color: #242424;
                border: 1px solid #333333;
                border-radius: 6px;
                margin: 5px;
                color: #e0e0e0;
            }
            QListWidget::item:selected {
                background-color: #2980b9;
                border: 1px solid #3498db;
                color: #ffffff;
            }
            QListWidget::item:hover {
                background-color: #333333;
                border: 1px solid #444444;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        
        # Search Box
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("Search by Player ID...")
        self.search_input.textChanged.connect(self.filter_items)
        layout.addWidget(self.search_input)
        
        # Image Grid List
        self.list_widget = QListWidget(self)
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setIconSize(QSize(100, 130))
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setSpacing(10)
        self.list_widget.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.list_widget)
        
        self.all_items = []
        self.load_images()

    def load_images(self):
        self.list_widget.clear()
        self.all_items = []
        
        if not os.path.exists(self.directory_path):
            return
            
        extensions = ('.jpg', '.jpeg', '.png')
        files = sorted([f for f in os.listdir(self.directory_path) if f.lower().endswith(extensions)])
        
        for file in files:
            full_path = os.path.join(self.directory_path, file)
            # Make a QListWidgetItem
            item = QListWidgetItem()
            item.setData(Qt.UserRole, full_path)
            
            # Text layout: ID
            # e.g., 0056_c1s2_000016_01.jpg
            pid = "Unknown"
            if "_" in file:
                pid = file.split("_")[0]
            item.setText(f"Player: {pid}")
            item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            item.setTextAlignment(Qt.AlignCenter)
            
            # Load thumbnail (lazy load or quick scaled)
            pixmap = QPixmap(full_path)
            thumbnail = pixmap.scaled(90, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            item.setIcon(QIcon(thumbnail))
            
            self.list_widget.addItem(item)
            self.all_items.append((pid, item))

    def filter_items(self, text):
        search_text = text.strip().lower()
        for pid, item in self.all_items:
            if not search_text or search_text in pid.lower():
                item.setHidden(False)
            else:
                item.setHidden(True)

    def on_item_double_clicked(self, item):
        img_path = item.data(Qt.UserRole)
        self.image_selected.emit(img_path)
