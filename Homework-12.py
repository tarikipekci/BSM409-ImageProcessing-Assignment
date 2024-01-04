import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_and_pepper_noise(image_path, salt_prob=0.02, pepper_prob=0.02):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Generate salt-and-pepper noise
    salt_mask = np.random.random(img.shape) < salt_prob
    pepper_mask = np.random.random(img.shape) < pepper_prob

    # Apply salt-and-pepper noise
    img[salt_mask] = 255
    img[pepper_mask] = 0

    # Display the results
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Image with Salt and Pepper Noise')
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')

    plt.show()

# Example usage
image_path = 'homeworks/Image/testImage.jpg'  # File path
add_salt_and_pepper_noise(image_path, salt_prob=0.02, pepper_prob=0.02)