import cv2, numpy as np

def read_images(filepath):
    img = cv2.imread(filepath)
    return img

def resize_images(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300)).astype(np.float32) / 255.0
    return img