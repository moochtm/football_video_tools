import torch
import cv2
import numpy

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

img2 = '1.png'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img2)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
pil_image = results.show()
open_cv_image = numpy.array(pil_image)
open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
# Convert RGB to BGR
# open_cv_image = open_cv_image[:, :, ::-1].copy()

cv2.imshow("Image", open_cv_image)

key = cv2.waitKey(1)
