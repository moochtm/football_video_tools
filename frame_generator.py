import cv2
import numpy
import os


def frames_from_images(input_path):
    left_path = os.path.join(input_path, "left")
    right_path = os.path.join(input_path, "right")
    __check_paths([left_path, right_path])
    
    left_images = sorted(os.listdir(left_path))
    right_images = sorted(os.listdir(right_path))

    image_count = len(left_images)
    if len(right_images) < image_count:
        image_count = len(right_images)

    for i in range(image_count):
        left_image = cv2.imread(os.path.join(left_path, left_images[i]))
        right_image = cv2.imread(os.path.join(right_path, right_images[i]))
        yield left_image, right_image


def __check_paths(paths):
    for path in paths:
        if not os.path.exists(path):
            raise NotADirectoryError
        if not os.path.isdir(path):
            raise NotADirectoryError

