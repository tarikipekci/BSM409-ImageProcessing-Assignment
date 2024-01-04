import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_esitleme(image_path):
    #Read Image
    img = cv2.imread(image_path, 0)

    equ = cv2.equalizeHist(img)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(' Equalized Histogram')
    plt.imshow(equ, cmap='gray')
    plt.axis('off')

    plt.show()

# Example
image_path = 'homeworks/Image/testImage.jpg'  # File path
histogram_esitleme(image_path)