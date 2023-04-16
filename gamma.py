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
from ex1_utils import LOAD_GRAY_SCALE
from cv2 import cv2
import numpy as np
title_window = 'Gamma Correction'
trackbar_name = 'Gamma:'
gamma_slider_max_val = 200
max_pix = 255
isColor = False
img = 0

import cv2


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    if rep == 1:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    gamma_slider_max_val = 200
    max_pix = 255
    title_window = 'Gamma Correction'
    trackbar_name = 'Gamma:'
    cv2.namedWindow(title_window)
    cv2.createTrackbar(trackbar_name, title_window, 100, gamma_slider_max_val,
                       lambda val: on_trackbar(val, img, max_pix))

    on_trackbar(100, img, max_pix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def on_trackbar(val: int, img: np.ndarray, max_pix: int):
    gamma = float(val) / 100
    inv_gamma = 1000 if gamma == 0 else 1.0 / gamma
    gammaMat = np.array([((i / float(max_pix)) ** inv_gamma) * max_pix
                         for i in np.arange(0, max_pix + 1)]).astype("uint8")
    img_after_gamma = cv2.LUT(img, gammaMat)

    cv2.imshow(title_window, img_after_gamma)

    pass


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
