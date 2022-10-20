
import numpy as np
import cv2

input_pts = np.float32([[720, 450], [1186, 452], [160, 994], [1738, 1014]])
output_pts = np.float32([[29, 289], [29, 101], [583, 268], [583, 101]])

# Compute the perspective transform M
M = cv2.getPerspectiveTransform(input_pts, output_pts)

to_convert = np.float32([[[952, 576]]])

# print(np.dot(M, to_convert))
converted = cv2.perspectiveTransform(to_convert, M)[0][0]
print(converted)
