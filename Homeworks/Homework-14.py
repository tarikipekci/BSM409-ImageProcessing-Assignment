import cv2
import numpy as np
import matplotlib.pyplot as plt

def color_segmentation(image_path, lower_threshold, upper_threshold):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert image to RGB (OpenCV reads images in BGR format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Define lower and upper thresholds for color segmentation
    lower_bound = np.array(lower_threshold, dtype=np.uint8)
    upper_bound = np.array(upper_threshold, dtype=np.uint8)

    # Create a mask for the specified color range
    mask = cv2.inRange(img_rgb, lower_bound, upper_bound)

    # Apply the mask to the original image
    segmented_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    # Display the results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Color Segmentation Mask')
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Segmented Image')
    plt.imshow(segmented_img)
    plt.axis('off')

    plt.show()

# Example usage
image_path = 'homeworks/Image/testImage.jpg'  # File path
lower_threshold = [0, 100, 0]  # Lower threshold for green color
upper_threshold = [50, 255, 50]  # Upper threshold for green color
color_segmentation(image_path, lower_threshold, upper_threshold)