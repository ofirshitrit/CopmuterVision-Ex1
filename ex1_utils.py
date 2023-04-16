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





def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):  # TODO
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # Check if the image is RGB or grayscale
    isRGB = len(imOrig.shape) == 3 and imOrig.shape[2] == 3

    # Convert to YIQ if image is RGB
    if isRGB:
        yiq = cv2.cvtColor(imOrig, cv2.COLOR_RGB2YIQ)
        img = yiq[:, :, 0]
    else:
        img = imOrig

    # Calculate initial segment borders
    segment_size = 256 // nQuant
    segment_borders = np.arange(0, 256, segment_size)
    segment_borders[-1] = 255

    # Iterate nIter times
    img_lst = []
    err_lst = []
    for i in range(nIter):
        # Calculate segment values
        segment_values = []
        for j in range(nQuant):
            segment = img[(segment_borders[j] <= img) & (img < segment_borders[j + 1])]
            segment_values.append(np.mean(segment))

        # Calculate new segment borders
        new_segment_borders = np.zeros(nQuant + 1)
        new_segment_borders[0] = 0
        new_segment_borders[-1] = 255
        for j in range(1, nQuant):
            new_segment_borders[j] = (segment_values[j - 1] + segment_values[j]) // 2

        # Quantize the image
        q_img = np.interp(img, new_segment_borders, segment_values)

        # Calculate MSE error
        mse = np.mean(np.square(img - q_img))

        # Update segment borders
        segment_borders = new_segment_borders

        # Append results to lists
        img_lst.append(q_img)
        err_lst.append(mse)

    # Convert back to RGB if image was RGB
    if isRGB:
        yiq[:, :, 0] = img_lst[-1]
        img_lst[-1] = cv2.cvtColor(yiq, cv2.COLOR_YIQ2RGB)

    # Plot error as function of iteration number
    plt.plot(err_lst)
    plt.title("MSE Error vs. Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Error")
    plt.show()

    # Return results
    return img_lst, err_lst

    pass
