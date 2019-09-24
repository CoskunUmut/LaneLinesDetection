import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import helpers

images = os.listdir("test_images/")

# Load Image
path = images[1]
img1 = cv2.imread("test_images/"+path)


# 2 Grayscale Image
gray1 = helpers.grayscale(img1)
result = gray1

# 3 Canny
canny1 = helpers.canny(result, 200, 350)
result = canny1


# 1 Region of Interest
vertices = np.array([((50, 540), (440, 325), (540, 325), (950, 540))])
roi1 = helpers.region_of_interest(result, vertices)
result = roi1

# 4 Hough Transformation
hough1 = helpers.hough_lines(result, 1, np.pi / 180, 25, 25, 200)
result = hough1

result = helpers.weighted_img(result, img1, 1.0, 1.0)

# Show result
cv2.imshow("canny", canny1)
# cv2.imshow("glbur", gblur1)
#cv2.imshow("roi", roi1)
cv2.imshow("hough", hough1)
cv2.imshow("Lane Line Detection", result)


if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
