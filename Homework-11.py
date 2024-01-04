import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_gaussian_blur(image_path, blur_kernel_size=5):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)

    # Display the results
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Gaussian Blur ({}x{})'.format(blur_kernel_size, blur_kernel_size))
    plt.imshow(blurred_img, cmap='gray')
    plt.axis('off')

    plt.show()

# Example usage
image_path = 'homeworks/Image/testImage.jpg'  # File path
apply_gaussian_blur(image_path, blur_kernel_size=5)