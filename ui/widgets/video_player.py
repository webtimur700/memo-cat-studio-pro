import cv2
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy
from PySide6.QtGui import QImage, QPixmap, QColor
from PySide6.QtCore import QTimer, Slot, Signal, Qt
import os

class VideoPlayer(QWidget):
    frame_updated = Signal(QPixmap)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #000;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: #666;")
        self.label.setText("Video Preview (Container Mode)")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)
        self.is_playing = False

    def load_video(self, path: str):
        self.stop()
        if not os.path.exists(path):
            self.label.setText(f"File not found: {path}")
            return
            
        # Используем CV2 для чтения, так как он работает в контейнере стабильнее GStreamer
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.label.setText("Failed to open video (Codec missing?)")
            return
            
        self.is_playing = True
        self.timer.start(33) # ~30 FPS

    @Slot()
    def _next_frame(self):
        if not self.cap or not self.is_playing:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop
            return
            
        # Convert BGR (OpenCV) to RGB (Qt)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale to fit
        scaled_pixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)

    def play(self):
        if self.cap:
            self.is_playing = True
            self.timer.start(33)

    def stop(self):
        self.is_playing = False
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.label.clear()
        self.label.setText("Stopped")

    def closeEvent(self, event):
        self.stop()
        super().closeEvent(event)
