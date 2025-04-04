import sys
import random
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor

class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.highlighted_box = None
        self.highlight_timer = QTimer()
        self.highlight_timer.timeout.connect(self.clear_highlight)
        # Warna awal
        self.current_color = QColor(0, 255, 0)  # Hijau sebagai default
        
    def init_ui(self):
        # Set window flags
        self.setWindowFlags(
            Qt.FramelessWindowHint |   # No frame
            Qt.WindowStaysOnTopHint |  # Always on top
            Qt.Tool                    # Not in taskbar
        )
        # Set window opacity
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Set to full screen size
        self.setGeometry(0, 0, self.get_screen_width(), self.get_screen_height())
        
        # Show the window
        self.show()
    
    def get_screen_width(self):
        # Get screen width
        return self.screen().size().width()
    
    def get_screen_height(self):
        # Get screen height
        return self.screen().size().height()
    
    def generate_random_color(self):
        # Generate random color (excluding white and very light colors)
        while True:
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            
            # Hindari warna putih dan warna sangat terang
            # (Jumlah RGB < 600 untuk menghindari warna terang)
            if r + g + b < 600:
                return QColor(r, g, b)
    
    def highlight_answer(self, box_coords):
        # Set the box to highlight
        self.highlighted_box = box_coords
        
        # Generate new random color for this prediction
        self.current_color = self.generate_random_color()
        
        # Repaint the widget
        self.repaint()
        
        # Set timer to clear highlight after 500ms
        self.highlight_timer.start(500)
    
    def clear_highlight(self):
        # Clear the highlighted box
        self.highlighted_box = None
        
        # Repaint the widget
        self.repaint()
        
    def paintEvent(self, event):
        if not self.highlighted_box:
            return
            
        # Create painter
        painter = QPainter(self)
        
        # Set pen properties
        pen = QPen(self.current_color)  # Gunakan warna saat ini
        pen.setWidth(3)
        painter.setPen(pen)
        
        # Draw rectangle around the answer
        x1, y1, x2, y2 = self.highlighted_box
        painter.drawRect(x1, y1, x2-x1, y2-y1)
        
    def close(self):
        # Stop timer if running
        if self.highlight_timer.isActive():
            self.highlight_timer.stop()
            
        # Call parent's close method
        super().close()