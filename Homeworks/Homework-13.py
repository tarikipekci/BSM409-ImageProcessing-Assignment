import cv2
import numpy as np
import matplotlib.pyplot as plt

def contraharmonic_mean_filter(image_path, filter_size=3, Q=1.5):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Pad the image to handle filter near borders
    padded_img = cv2.copyMakeBorder(img, filter_size // 2, filter_size // 2, filter_size // 2, filter_size // 2, cv2.BORDER_REFLECT)

    # Apply contraharmonic mean filter
    result_img = np.zeros_like(img, dtype=np.float64)
    for i in range(filter_size, padded_img.shape[0] - filter_size):
        for j in range(filter_size, padded_img.shape[1] - filter_size):
            neighborhood = padded_img[i - filter_size // 2:i + filter_size // 2 + 1, j - filter_size // 2:j + filter_size // 2 + 1]
            numerator = np.sum(neighborhood**(Q + 1))
            denominator = np.sum(neighborhood**Q)
            result_img[i - filter_size // 2, j - filter_size // 2] = numerator / max(denominator, 1e-10)

    result_img = np.clip(result_img, 0, 255).astype(np.uint8)

    # Display the results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Contraharmonic Mean Filter')
    plt.imshow(result_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference Image')
    plt.imshow(np.abs(img - result_img), cmap='gray')
    plt.axis('off')

    plt.show()

# Example usage
image_path = 'homeworks/Image/testImage.jpg'  # File path
contraharmonic_mean_filter(image_path, filter_size=3, Q=1.5)