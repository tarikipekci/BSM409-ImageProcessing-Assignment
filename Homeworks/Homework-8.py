import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_sobel_filter(image_path):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Sobel filter
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude and direction
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x)

    # Display the results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Sobel X')
    plt.imshow(np.abs(sobel_x), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Sobel Y')
    plt.imshow(np.abs(sobel_y), cmap='gray')
    plt.axis('off')

    plt.show()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.title('Gradient Magnitude')
    plt.imshow(gradient_magnitude, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Gradient Direction')
    plt.imshow(gradient_direction, cmap='jet')
    plt.axis('off')

    plt.show()

# Example usage
image_path = 'homeworks/Image/testImage.jpg'  # File path
apply_sobel_filter(image_path)