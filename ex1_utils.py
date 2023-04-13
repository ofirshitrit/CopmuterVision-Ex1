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

    if representation == 1:
        plt.imshow(img, cmap='gray')
        plt.show()
    else:
        plt.imshow(img)
        plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray: #TODO
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    RGB2YIQ_mat = np.array([0.299, 0.587, 0.114, 0.596, -0.275, -0.321, 0.212, -0.523, 0.311]).reshape(3, 3)
    shapeImg = imgRGB.shape
    imgRGB = imgRGB.reshape(-1, 3)
    YIQ_img = imgRGB.dot(RGB2YIQ_mat).reshape(shapeImg)
    return YIQ_img



def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray: #TODO
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray): #TODO
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]): #TODO
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
