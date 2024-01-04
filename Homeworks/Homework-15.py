import cv2
import numpy as np
import matplotlib.pyplot as plt

def morphological_operations(image_path, kernel_size=3, opening_iterations=1, closing_iterations=1):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Create a rectangular kernel for morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform morphological opening
    opened_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=opening_iterations)

    # Perform morphological closing
    closed_img = cv2.morphologyEx(opened_img, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)

    # Display the results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Morphological Opening')
    plt.imshow(opened_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Morphological Closing')
    plt.imshow(closed_img, cmap='gray')
    plt.axis('off')

    plt.show()

# Example usage
image_path = 'homeworks/Image/testImage.jpg'  # File path
morphological_operations(image_path, kernel_size=5, opening_iterations=2, closing_iterations=2)