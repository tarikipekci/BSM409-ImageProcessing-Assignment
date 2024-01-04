import cv2
import numpy as np
import matplotlib.pyplot as plt

def mean_filter(image_path, filter_size=3):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if filter_size % 2 == 0:
        raise ValueError("Must be odd")

    kernel = np.ones((filter_size, filter_size), np.float32) / (filter_size ** 2)

    filtered_img = cv2.filter2D(img, -1, kernel)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Mean filter ({}x{})'.format(filter_size, filter_size))
    plt.imshow(filtered_img, cmap='gray')
    plt.axis('off')

    plt.show()

image_path = 'homeworks/Image/testImage.jpg'  # File path
mean_filter(image_path, filter_size=3)