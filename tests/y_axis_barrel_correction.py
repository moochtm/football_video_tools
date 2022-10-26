
import cv2 as cv
import numpy as np
import math

print(1 + 1 * 2)

f_img = "/Users/home/Code/football_video_tools/tests/UnderstandingLensDistortion/output2 25Mbps/output2_0010.png"
im_cv = cv.imread(f_img)

# grab the dimensions of the image
(h, w, _) = im_cv.shape

scale_y = 2
center_y = h/2
scale_x = 2
center_x = w/2
radius = 2000
amount = 1

# set up the x and y maps as float32
flex_x = np.zeros((h, w), np.float32)
flex_y = np.zeros((h, w), np.float32)

# create map with the barrel pincushion distortion formula
# for y in range(h):
#     delta_y = scale_y * (y - center_y)
#     if int(y/100) == y/100:
#         print(y)
#     for x in range(w):
#         # determine if pixel is within an ellipse
#         delta_x = scale_x * (x - center_x)
#         distance = delta_x * delta_x + delta_y * delta_y
#         if distance >= (radius * radius):
#             flex_x[y, x] = x
#             flex_y[y, x] = y
#         else:
#             factor = 1.0
#             if distance > 0.0:
#                 factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), -amount)
#             flex_x[y, x] = factor * delta_x / scale_x + center_x
#             flex_y[y, x] = factor * delta_y / scale_y + center_y

k1 = -0.000000000227
k2 = 0 # -0.00000000000022

for y in range(h):
    if int(y/100) == y/100:
        print(y)
    for x in range(w):
        r = math.sqrt(((y - center_y) ** 2) + ((x - center_x) ** 2))
        k1_r2 = k1 * (r ** 2)
        k2_r4 = k2 * (r ** 4)
        r_sq = r ** 2
        flex_x[y, x] = x + (x - center_x) * (1 + k1_r2 + k2_r4)
        flex_y[y, x] = y + (y - center_y) * (1 + k1_r2 + k2_r4)

# do the remap  this is where the magic happens
dst = cv.remap(im_cv, flex_x, flex_y, cv.INTER_LINEAR)

cv.imshow('src', im_cv)
cv.waitKey(0)
cv.imshow('dst', dst)

cv.waitKey(0)
cv.destroyAllWindows()