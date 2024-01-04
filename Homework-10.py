import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_smoothing_and_sharpening(image_path, blur_kernel_size=5, sharpening_strength=1.5):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur for smoothing
    smoothed_img = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)

    # Apply Laplacian operator for sharpening
    laplacian_img = cv2.Laplacian(smoothed_img, cv2.CV_64F)

    # Combine the original and sharpened images
    sharpened_img = img + sharpening_strength * laplacian_img

    # Clip values to stay within valid image intensity range
    sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)

    # Display the results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title('Smoothed Image (Gaussian Blur)')
    plt.imshow(smoothed_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title('Laplacian of Smoothed Image (Sharpening)')
    plt.imshow(laplacian_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title('Sharpened Image')
    plt.imshow(sharpened_img, cmap='gray')
    plt.axis('off')

    plt.show()

# Example usage
image_path = 'homeworks/Image/testImage.jpg'  # File path
apply_smoothing_and_sharpening(image_path, blur_kernel_size=5, sharpening_strength=1.5)