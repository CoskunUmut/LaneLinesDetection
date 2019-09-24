import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import helpers

images = os.listdir("test_images/")

# Load Image
img1 = images[0]
img1 = mpimg.imread('test_images/solidWhiteRight.jpg')


# 2 Grayscale Image
gray1 = helpers.grayscale(img1)
result = gray1

# 3 Canny
canny1 = helpers.canny(result, 250, 400)
result = canny1

# 1 Region of Interest
vertices = np.array([((50, 540), (440, 300), (540, 300), (950, 540))])
roi1 = helpers.region_of_interest(result, vertices)
result = roi1

# 4 Hough Transformation
hough1 = helpers.hough_lines(result, 1, np.pi / 180, 50, 50, 500)

# # 3 Gaussian Blur
# gblur1 = helpers.gaussian_blur(result, 5)
# result = gblur1

# Show result
#cv2.imshow("canny", canny1)
# cv2.imshow("glbur", gblur1)
cv2.imshow("roi", roi1)
cv2.imshow("hough", hough1)


if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
