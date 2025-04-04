import cv2
import numpy as np

def preprocess_image(image):
    """Convert image to grayscale and apply threshold"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return binary

def calculate_image_hash(image):
    """Calculate perceptual hash of an image"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.img_hash.pHash(gray)[0]

def compare_images(img1, img2):
    """Compare two images using pixel-wise difference"""
    # Resize to same dimensions
    resized1 = cv2.resize(img1, (32, 32))
    resized2 = cv2.resize(img2, (32, 32))
    
    # Calculate difference
    diff = np.sum(np.abs(resized1 - resized2)) / (32*32*255)
    return diff