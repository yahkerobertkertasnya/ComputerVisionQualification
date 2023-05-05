import cv2
import numpy as np



def gray_image(image=None):
    ocv_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    avg_gray = np.dot(image.copy(), [1/3, 1/3, 1/3])

    maxcha = max(np.max(image[:, :, 0]), np.max(image[:, :, 1]), np.max(image[:, :, 2]))
    mincha = min(np.min(image[:, :, 0]), np.min(image[:, :, 1]), np.min(image[:, :, 2]))

    lig_gray = np.dot(image.copy(), [(maxcha + mincha) / 2, (maxcha + mincha) / 2, (maxcha + mincha) / 2])
    lum_gray = np.dot(image.copy(), [0.07, 0.71, 0.21])
    wavg_gray = np.dot(image.copy(), [0.114, 0.587, 0.299])

    gray_labels = ['ocv', 'avg', 'lig', 'lum', 'wag']
    gray_images = [ocv_gray, avg_gray, lig_gray, lum_gray, wavg_gray]

    return 1, 5, zip(gray_labels, gray_images)

def manual_mean_filter(source, ksize, height, width):
    np_source = np.array(source)
    for i in range(height - ksize):
        for j in range(width - ksize):
            matrix = np.array(np_source[i: (i + ksize), j: (j + ksize)]).flatten()
            mean = np.mean(matrix)
            np_source[i + ksize // 2, j + ksize // 2] = mean
    return np_source

def manual_median_filter(source, ksize, height, width):
    np_source = np.array(source)
    for i in range(height - ksize):
        for j in range(width - ksize):
            matrix = np.array(np_source[i: (i + ksize), j: (j + ksize)]).flatten()
            median = np.median(matrix)
            np_source[i + ksize // 2, j + ksize // 2] = median
    return np_source

def blur_image(image=None):
    height, width = image.shape[:2]

    b, g, r = cv2.split(image)

    manmean_blur = cv2.merge((manual_mean_filter(b, 3, height, width), manual_mean_filter(g, 3, height, width), manual_mean_filter(r, 3, height, width)))
    manmed_blur = cv2.merge((manual_median_filter(b, 3, height, width), manual_median_filter(g, 3, height, width), manual_median_filter(r, 3, height, width)))

    cv2_blur = cv2.blur(image.copy(), (3, 3))
    median_blur = cv2.medianBlur(image.copy(), 3)
    gauss_blur = cv2.GaussianBlur(image.copy(), (3, 3), 2.0)
    bilateral_blur = cv2.bilateralFilter(image.copy(), 3, 150, 150)
    blur_labels = ['cv2', 'median', 'gauss', 'bilateral', 'merged mean', 'merged median']
    blur_images = [cv2.cvtColor(cv2_blur, cv2.COLOR_BGR2RGB), cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB),
                  cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2RGB), cv2.cvtColor(gauss_blur, cv2.COLOR_BGR2RGB),
                  cv2.cvtColor(bilateral_blur, cv2.COLOR_BGR2RGB), cv2.cvtColor(manmean_blur, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(manmed_blur, cv2.COLOR_BGR2RGB)]

    return 2, 3, zip(blur_labels, blur_images)


def thresh_image(image=None):
    man_thresh = image.copy()
    thresh = 100
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            if man_thresh[i, j] > thresh:
                man_thresh[i, j] = 255
            else:
                man_thresh[i, j] = 0

    _, bin_thresh = cv2.threshold(image.copy(), 100, 255, cv2.THRESH_BINARY)
    _, binv_thresh = cv2.threshold(image.copy(), 100, 255, cv2.THRESH_BINARY_INV)
    _, mask_thresh = cv2.threshold(image.copy(), 100, 255, cv2.THRESH_MASK)
    _, otsu_thresh = cv2.threshold(image.copy(), 100, 255, cv2.THRESH_OTSU)
    _, to_thresh = cv2.threshold(image.copy(), 100, 255, cv2.THRESH_TOZERO)
    _, tinv_thresh = cv2.threshold(image.copy(), 100, 255, cv2.THRESH_TOZERO_INV)
    _, tri_thresh = cv2.threshold(image.copy(), 100, 255, cv2.THRESH_TRIANGLE)
    _, trunc_thresh = cv2.threshold(image.copy(), 100, 255, cv2.THRESH_TRUNC)

    thresh_images = [man_thresh, bin_thresh, binv_thresh, mask_thresh, otsu_thresh, to_thresh, tinv_thresh, tri_thresh, trunc_thresh]
    thresh_labels = ["Manual", "Binary", "Binary Inverse", "Mask", "Otsu", "ToZero", "ToZero Inverse", "Triangle", "Truncate"]

    return 3, 3, zip(thresh_labels, thresh_images)