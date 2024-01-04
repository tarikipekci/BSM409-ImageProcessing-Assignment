import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_filter(image_path, kernel_size=5, sigma=1.0):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Gaussian Filter ({}x{}, Sigma={})'.format(kernel_size, kernel_size, sigma))
    plt.imshow(blurred_img, cmap='gray')
    plt.axis('off')

    plt.show()

image_path = 'homeworks/Image/testImage.jpg'  # File path
gaussian_filter(image_path, kernel_size=5, sigma=1.0)