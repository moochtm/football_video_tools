import cv2
import numpy as np
import argparse
import time

from frame_generator import frames_from_images
from lens_distortion_corrector import correct_distortion
from frame_stitcher import Stitcher

parser = argparse.ArgumentParser()
parser.add_argument('--images', help="List of image file paths", type=str)
parser.add_argument('--left_lens_config', type=str)
parser.add_argument('--right_lens_config', type=str)
parser.add_argument('--videos', help="List of video file paths", type=str)
parser.add_argument('--verbose', help="To print statements", default=True)
args = parser.parse_args()


if __name__ == '__main__':
    images_path = args.images
    print(args)
    stitcher = Stitcher()

    for left_img, right_img in frames_from_images(images_path):
        left_img = correct_distortion(left_img, args.left_lens_config)
        #cv2.imshow("undistorted image", left_img)
        #cv2.waitKey(0)

        right_img = correct_distortion(right_img, args.right_lens_config)
        #cv2.imshow("undistorted image", right_img)
        #cv2.waitKey(0)

        img = stitcher.stitch([left_img, right_img])
        cv2.imshow("undistorted image", img)
        cv2.waitKey(0)



