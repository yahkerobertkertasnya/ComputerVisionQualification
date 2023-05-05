import cv2
import numpy as np


def laplacian(image=None):
    lap_8u = cv2.Laplacian(image, cv2.CV_8U)
    lap_16s = cv2.Laplacian(image, cv2.CV_16S)
    lap_64f = cv2.Laplacian(image, cv2.CV_64F)

    laplace_labels = ['8U', '16S', '64F']
    laplace_images = [lap_8u, lap_16s, lap_64f]

    return 1, 3, zip(laplace_labels, laplace_images)


def sobel(image=None):
    sobel_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, 3)
    sobel_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, 3)

    merged_sobel = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    merged_sobel *= 255 / merged_sobel.max()

    sobel_labels = ['x', 'y', 'merged']
    sobel_images = [sobel_x, sobel_y, merged_sobel]

    return 1, 3, zip(sobel_labels, sobel_images)


def canny(image=None):
    canny_50100 = cv2.Canny(image, 50, 100)
    canny_100150 = cv2.Canny(image, 100, 150)
    canny_150200 = cv2.Canny(image, 150, 200)

    canny_labels = ['50-100', '100-150', '150-200']
    canny_images = [canny_50100, canny_100150, canny_150200]

    return 1, 3, zip(canny_labels, canny_images)

