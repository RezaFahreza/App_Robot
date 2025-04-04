import numpy as np
from mss import mss

class ScreenCapture:
    def __init__(self):
        self.sct = mss()
        
        # By default, capture primary monitor
        self.monitor = self.sct.monitors[1]  # monitors[0] is all monitors combined
        
        # Print monitor information
        print(f"Monitor dimensions: {self.monitor}")
    
    def capture(self):
        # Capture screen
        screenshot = np.array(self.sct.grab(self.monitor))
        
        # Get and print screenshot dimensions
        height, width = screenshot.shape[:2]
        print(f"Screenshot dimensions: {width}x{height} pixels")
        
        # Convert BGRA to BGR (remove alpha channel)
        if screenshot.shape[2] == 4:
            screenshot = screenshot[:, :, :3]
            
        return screenshot

# # Test the screen capture
# if __name__ == "__main__":
#     import cv2
    
#     sc = ScreenCapture()
#     img = sc.capture()
    
#     # Display the dimensions
#     height, width = img.shape[:2]
#     print(f"Final image dimensions: {width}x{height} pixels")
    
#     # Display the captured image
#     cv2.imshow("Screen Capture Test", img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()