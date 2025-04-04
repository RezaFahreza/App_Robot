import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QHBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from src.bot import Bot

class BotThread(QThread):
    status_signal = pyqtSignal(str)
    
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.bot = None
        self.running = False
        
    def run(self):
        try:
            self.bot = Bot(self.model_path)
            self.running = True
            self.status_signal.emit("Bot started")
            
            while self.running:
                try:
                    self.bot.process_frame()
                except Exception as e:
                    # Log error tapi tetap lanjutkan
                    print(f"Error in processing frame: {str(e)}")
                    self.status_signal.emit(f"Error: {str(e)}")
                    # Jeda sejenak sebelum mencoba lagi
                    self.msleep(500)
                
                # Slow down the loop slightly
                self.msleep(30)  # ~33 fps
                
            self.bot.cleanup()
            self.status_signal.emit("Bot stopped")
        except Exception as e:
            # Error fatal yang menyebabkan bot harus berhenti
            print(f"Fatal error: {str(e)}")
            self.status_signal.emit(f"Bot crashed: {str(e)}")
            self.running = False
    
    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.bot_thread = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("Sikap Kerja Bot")
        self.setGeometry(100, 100, 400, 200)
        
        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Status label
        self.status_label = QLabel("Bot is not running")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # Buttons layout
        button_layout = QHBoxLayout()
        
        # Start button
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_bot)
        button_layout.addWidget(self.start_button)
        
        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_bot)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        main_layout.addLayout(button_layout)
        
    def start_bot(self):
        if not self.bot_thread:
            model_path = os.path.join("models", "yolo_model.pt")
            self.bot_thread = BotThread(model_path)
            self.bot_thread.status_signal.connect(self.update_status)
            self.bot_thread.start()
            
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
    
    def stop_bot(self):
        if self.bot_thread and self.bot_thread.isRunning():
            self.bot_thread.stop()
            self.bot_thread = None
            
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def update_status(self, status):
        self.status_label.setText(status)
    
    def closeEvent(self, event):
        self.stop_bot()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())