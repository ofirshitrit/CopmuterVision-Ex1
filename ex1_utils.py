"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import sys
from typing import List
import cv2

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int_:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 324249150


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # Load the image using OpenCV
    img = cv2.imread(filename)

    # Convert the image to grayscale or RGB
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif representation == 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize the pixel intensities to the range [0,1]
    img = img.astype(np.float_) / 255.0

    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    # Read the image using imReadAndConvert function
    img = imReadAndConvert(filename, representation)

    # Create a new figure window
    plt.figure()

    # Display the image using plt.imshow function
    if representation == 1:
        plt.imshow(img, cmap='gray')
    elif representation == 2:
        plt.imshow(img)
    else:
        print("Invalid representation value! Please enter 1 for grayscale or 2 for RGB.")
        return

    # Show the plot
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # Convert the RGB image to YIQ using the transformation matrix
    T = np.array([[0.299, 0.587, 0.114],
                  [0.596, -0.275, -0.321],
                  [0.212, -0.523, 0.311]])
    imYIQ = np.dot(imgRGB, T.T)
    return imYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:  # TODO
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """

    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                               [0.596, -0.275, -0.321],
                               [0.212, -0.523, 0.311]])


    OrigShape = imgYIQ.shape
    yiq2rgb = np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(yiq_from_rgb).transpose()).reshape(OrigShape)
    return yiq2rgb
    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """

    isColored = False
    YIQimg = 0
    tmpMat = imgOrig
    if len(imgOrig.shape) == 3:  # it's RGB convert to YIQ and take the Y dimension
        YIQimg = transformRGB2YIQ(imgOrig)
        tmpMat = YIQimg[:, :, 0]
        isColored = True
    tmpMat = cv2.normalize(tmpMat, None, 0, 255, cv2.NORM_MINMAX)
    tmpMat = tmpMat.astype('uint8')
    histOrg = np.histogram(tmpMat.flatten(), bins=256)[0]  # original image histogram
    cumSum = np.cumsum(histOrg)  # image cumSum

    LUT = np.ceil((cumSum / cumSum.max()) * 255)  # calculate the LUT table
    imEqualized = tmpMat.copy()
    for i in range(256):  # give the right value for each pixel according to the LUT table
        imEqualized[tmpMat == i] = int(LUT[i])

    histEq = np.histogram(imEqualized.flatten().astype('uint8'), bins=256)[0]  # equalized image histogram

    imEqualized = imEqualized / 255
    if isColored:  # RGB img -> convert back to RGB color space
        YIQimg[:, :, 0] = imEqualized
        imEqualized = transformYIQ2RGB(YIQimg)

    return imEqualized, histOrg, histEq


def case_RGB(imgOrig: np.ndarray) -> (bool, np.ndarray, np.ndarray):
    isRGB = imgOrig.ndim == 3 and imgOrig.shape[-1] == 3
    if isRGB:
        imgYIQ = transformRGB2YIQ(imgOrig)
        imgOrig = imgYIQ[..., 0]
        return True, imgYIQ, imgOrig
    return False, None, imgOrig


def back_to_rgb(yiq_img: np.ndarray, y_to_update: np.ndarray) -> np.ndarray:
    yiq_img[:, :, 0] = y_to_update
    return transformYIQ2RGB(yiq_img)


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to *nQuant* colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    isRGB, yiq_img, imOrig = case_RGB(imOrig)

    if np.amax(imOrig) <= 1:  # picture is normalized
        imOrig = (imOrig * 255).astype('uint8')

    histOrg, bin_edges = np.histogram(imOrig, 256, [0, 255])

    z = np.linspace(0, 255, nQuant + 1, dtype=int)  # boundaries
    q = np.zeros(nQuant)

    qImage_list = []
    error_list = []

    for i in range(nIter):
        new_img = np.zeros(imOrig.shape)

        for cell in range(len(q)):
            # Determine the range of pixel intensities for this cell
            left = z[cell]
            right = z[cell + 1]
            cell_range = np.arange(left, right)

            # Compute the average intensity for this cell, weighted by the pixel counts in its range
            hist_cell = histOrg[left:right]
            weights = hist_cell / np.sum(hist_cell)
            q[cell] = np.sum(weights * cell_range)

            # Assign the average intensity to all pixels within the cell's range
            condition = np.logical_and(imOrig >= left, imOrig < right)
            new_img[condition] = q[cell]

        MSE = mean_squared_error(imOrig / 255, new_img / 255)
        error_list.append(MSE)

        if isRGB:
            new_img = back_to_rgb(yiq_img, new_img / 255)

        qImage_list.append(new_img)
        z[1:-1] = (q[:-1] + q[1:]) / 2

        if len(error_list) >= 2 and abs(
                error_list[-1] - error_list[-2]) <= sys.float_info.epsilon:  # check if converged
            break

    return qImage_list, error_list


