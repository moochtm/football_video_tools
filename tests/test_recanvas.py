import sys
from random import randint
import cv2 as cv


def pad(src, top=0, bottom=0, left=0, right=0, color=(0, 0, 0, 0)):
    borderType = cv.BORDER_CONSTANT
    window_name = "copyMakeBorder Demo"

    # Check if image is loaded fine
    if src is None:
        print("Error opening image!")
        print("Usage: copy_make_border.py [image_name -- default lena.jpg] \n")
        return -1

    print(
        "\n"
        "\t   copyMakeBorder Demo: \n"
        "     -------------------- \n"
        " ** Press 'c' to set the border to a random constant value \n"
        " ** Press 'r' to set the border to be replicated \n"
        " ** Press 'ESC' to exit the program "
    )

    dst = cv.copyMakeBorder(
        src, int(top), int(bottom), int(left), int(right), borderType, None, color
    )

    return dst
