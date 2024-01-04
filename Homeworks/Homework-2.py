import cv2
import numpy as np
import matplotlib.pyplot as plt


def bit_plane_slice(image, bit):
    gray_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    bit_plane = (gray_image >> bit) & 1

    return bit_plane


def display_bit_plane(image, bit):
    bit_plane = bit_plane_slice(image, bit)

    plt.imshow(bit_plane, cmap='gray')
    plt.title(f'Bit Plane {bit}')
    plt.axis('off')
    plt.show()


image_path = 'homeworks/Image/testImage.jpg'
bit_position = 7
display_bit_plane(image_path, bit_position)
