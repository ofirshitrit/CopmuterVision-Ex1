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
    img = cv2.imread(filename)
    if img is None:
        sys.exit("Could not read the image.")

    # If the image is grayscale, the shape will have only 2 dimensions.
    imgIsGray = len(img.shape) == 2

    # If the image is RGB, the shape will have 3 dimensions and the last dimension will have a size of 3
    imgIsRGB = len(img.shape) == 3 and img.shape[2] == 3

    if imgIsGray and representation == LOAD_RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif imgIsRGB and representation == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # normalize pixel intensities to [0, 1]
    img = img.astype(np.float_) / 255.0

    # return the image as a numpy array
    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)

    # Display the image
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


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

    conversion_mat = np.array([[0.299, 0.587, 0.114],
                               [0.596, -0.275, -0.321],
                               [0.212, -0.523, 0.311]])

    # get dimensions of input YIQ image
    height, width, _ = imgYIQ.shape

    # reshape input YIQ image to (height*width)x3 matrix
    imYIQ = imgYIQ.reshape((-1, 3))

    # convert YIQ to RGB using matrix multiplication
    imRGB = np.dot(imYIQ, conversion_mat.T)

    # reshape output RGB image to height x width x 3
    imRGB = imRGB.reshape((height, width, 3))

    return imRGB
    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):  # TODO
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """

    # check if image is RGB
    is_rgb = len(imgOrig.shape) == 3 and imgOrig.shape[2] == 3

    # convert RGB to YIQ and equalize Y channel
    if is_rgb:
        yiq_img = transformRGB2YIQ(imgOrig)
        y_channel = yiq_img[:, :, 0]
        y_hist, bins = np.histogram(y_channel.flatten(), 256, [0, 256])
        y_cumsum = np.cumsum(y_hist)
        y_lut = np.uint8(255 * y_cumsum / y_cumsum[-1])
        y_channel_eq = y_lut[y_channel]
        yiq_img[:, :, 0] = y_channel_eq
        img_eq = transformYIQ2RGB(yiq_img)
    else:
        img_eq = np.copy(imgOrig)
        hist, bins = np.histogram(img_eq.flatten(), 256, [0, 256])
        cumsum = np.cumsum(hist)
        lut = np.uint8(255 * cumsum / cumsum[-1])
        img_eq = lut[img_eq]

    # compute histograms of original and equalized images
    hist_org, bins = np.histogram(imgOrig.flatten(), 256, [0, 256])
    hist_eq, bins = np.histogram(img_eq.flatten(), 256, [0, 256])

    # normalize output image
    img_eq = img_eq.astype(np.float) / 255.0

    return img_eq, hist_org, hist_eq
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):  # TODO
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
