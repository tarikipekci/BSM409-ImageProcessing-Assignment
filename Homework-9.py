import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_blur_and_laplacian(image_path, blur_kernel_size=5):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)

    # Apply Laplacian operator
    laplacian_img = cv2.Laplacian(blurred_img, cv2.CV_64F)

    # Display the results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Blurred Image')
    plt.imshow(blurred_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Laplacian of Blurred Image')
    plt.imshow(laplacian_img, cmap='gray')
    plt.axis('off')

    plt.show()

# Example usage
image_path = 'homeworks/Image/testImage.jpg'  # File path
apply_blur_and_laplacian(image_path, blur_kernel_size=5)