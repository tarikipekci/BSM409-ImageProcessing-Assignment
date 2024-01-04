import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(image_path, min_output=0, max_output=255):

    img = cv2.imread(image_path, 0)

    min_input = np.min(img)
    max_input = np.max(img)

    stretched_img = (img - min_input) * ((max_output - min_output) / (max_input - min_input)) + min_output
    stretched_img = np.clip(stretched_img, min_output, max_output).astype(np.uint8)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('contrast stretching')
    plt.imshow(stretched_img, cmap='gray')
    plt.axis('off')

    plt.show()

image_path = 'homeworks/Image/testImage.jpg'
contrast_stretching(image_path)