import cv2
import numpy as np
import matplotlib.pyplot as plt

def gamma_correction(image_path, gamma=1.0):
    img = cv2.imread(image_path)

    gamma_corrected = np.power(img / 255.0, gamma)
    gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Gamma correction (Gamma = {})'.format(gamma))
    plt.imshow(cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

image_path = 'homeworks/Image/testImage.jpg'  # File path
gamma_correction(image_path, gamma=1.5)