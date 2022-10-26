import cv2
import numpy as np

import math

image = cv2.imread('/Users/home/Code/football_video_tools/input/test_inputs/2/left/output001.png')

rows, cols = image.shape[:2]

# Vertical wave

img_output = np.zeros(image.shape, dtype=image.dtype)
for i in range(rows):
    for j in range(cols):
        offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
        offset_y = 0
        if j + offset_x < rows:
            img_output[i, j] = image[i, (j + offset_x) % cols]
        else:
            img_output[i, j] = 0

# Displaying the image.
cv2.imshow("distorted image", image)
cv2.waitKey(0)
cv2.imshow("distorted image", img_output)
cv2.waitKey(0)
