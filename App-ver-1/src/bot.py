from ultralytics import YOLO
import numpy as np
import cv2
import time
from .screen_capture import ScreenCapture
from .symbol_processor import SymbolProcessor
from .overlay import OverlayWindow

class Bot:
    def __init__(self, model_path):
        # Initialize YOLO model
        self.model = YOLO("models/best.pt")
        
        # Initialize screen capture
        self.screen_capture = ScreenCapture()
        
        # Get initial screen size to set model image size
        initial_screenshot = self.screen_capture.capture()
        self.screen_height, self.screen_width = initial_screenshot.shape[:2]
        print(f"Screen dimensions: {self.screen_width}x{self.screen_height}")
        
        # Initialize symbol processor
        self.symbol_processor = SymbolProcessor()
        
        # Initialize overlay window
        self.overlay = OverlayWindow()
        
        # Cache for reference symbols
        self.ref_symbols = None
        self.ref_hash = None
        self.column_index = 0
        
        # Performance tracking
        self.last_process_time = 0
        
    def detect_roi(self, screenshot):
        # Get current screenshot dimensions
        height, width = screenshot.shape[:2]
        
        # Check if screen dimensions have changed
        if height != self.screen_height or width != self.screen_width:
            self.screen_height, self.screen_width = height, width
            print(f"Screen dimensions updated: {width}x{height}")
        
        # Run inference with YOLO using dynamic image size
        results = self.model(
            screenshot, 
            conf=0.8,      # Confidence threshold
            imgsz=width,    # Use screen width to maintain aspect ratio
        )
        
        roi = {}
        
        # Process detection results
        for result in results:
            boxes = result.boxes  # Bounding boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get class id and convert to class name
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                
                # Map class to ROI
                if class_name == 'REFERENSI':
                    roi['reference'] = (x1, y1, x2, y2)
                elif class_name == 'SOAL':
                    roi['question'] = (x1, y1, x2, y2)
                elif class_name in ['A', 'B', 'C', 'D', 'E']:
                    roi[f'option_{class_name.lower()}'] = (x1, y1, x2, y2)
        
        return roi
    
    # Rest of the code remains the same
    # Fungsi process_frame perlu diperbarui

    def process_frame(self):
        try:
            start_time = time.time()
            
            # Capture screen
            screenshot = self.screen_capture.capture()
            
            # Detect ROIs
            roi = self.detect_roi(screenshot)
            
            # Check if all necessary ROIs are detected
            if 'reference' not in roi or 'question' not in roi:
                return
            
            # Extract reference area
            ref_area = screenshot[roi['reference'][1]:roi['reference'][3], 
                                roi['reference'][0]:roi['reference'][2]]
            
            # Gunakan metode hashing sederhana sebagai pengganti cv2.img_hash
            gray_ref = cv2.cvtColor(ref_area, cv2.COLOR_BGR2GRAY)
            # Resize untuk mengurangi variasi kecil
            small_ref = cv2.resize(gray_ref, (32, 32))
            # Flatten dan konversi ke byte array untuk perbandingan
            current_hash = small_ref.flatten()
            
            # If column changed or first run
            if self.ref_hash is None:
                self.ref_hash = current_hash
                self.ref_symbols = self.symbol_processor.extract_reference_symbols(ref_area)
                self.column_index += 1
                print(f"Column changed: {self.column_index}")
            else:
                # Bandingkan hash dengan menghitung perbedaan rata-rata
                hash_diff = np.mean(np.abs(self.ref_hash - current_hash))
                if hash_diff > 10.0:  # Threshold untuk perubahan signifikan
                    # Process reference symbols
                    self.ref_symbols = self.symbol_processor.extract_reference_symbols(ref_area)
                    self.ref_hash = current_hash
                    self.column_index += 1
                    print(f"Column changed: {self.column_index}, Difference: {hash_diff}")
            
            # Extract question area
            question_area = screenshot[roi['question'][1]:roi['question'][3], 
                                    roi['question'][0]:roi['question'][2]]
            
            # Process question symbols
            question_symbols = self.symbol_processor.extract_question_symbols(question_area)
            
            # Find missing symbol
            answer = self.symbol_processor.find_missing_symbol(self.ref_symbols, question_symbols)
            
            # If answer found, highlight it
            if answer and f'option_{answer.lower()}' in roi:
                answer_box = roi[f'option_{answer.lower()}']
                self.overlay.highlight_answer(answer_box)
                
                # Print processing time
                self.last_process_time = time.time() - start_time
                print(f"Answer: {answer}, Processing time: {self.last_process_time*1000:.1f}ms")
                
        except Exception as e:
            print(f"Error in process_frame: {str(e)}")
        
    def cleanup(self):
        # Clean up resources
        self.overlay.close()