import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import helpers
from sklearn.linear_model import LinearRegression


# images = os.listdir("test_images/")
# Load Image
# path = images[5]
# img1 = cv2.imread("test_images/"+path)
# cap = cv2.VideoCapture("test_videos/solidWhiteRight.mp4")
cap = cv2.VideoCapture("test_videos/challenge.mp4")
# cap = cv2.VideoCapture("test_videos/solidYellowLeft.mp4")
_, img1 = cap.read()
height, width, _ = img1.shape
last_result = []

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, img1 = cap.read()
    if not ret:
        break

    result = img1
    # 1.1 Thresholding
    _, result = cv2.threshold(result, 215, 255, cv2.THRESH_BINARY)

    # # 2 Canny
    canny1 = cv2.Canny(result, 75, 200)
    result = canny1

    vertices_right = []
    vertices_left = []
    # RoI Left
    vertices_left = np.array(
        [((int(width*0.1), int(height*0.9)),
          (int(width*0.45), int(height*0.59)),
            (int(width*0.55), int(height*0.59)),
            (int(width*0.35), int(height*0.9)))])
    result_left = helpers.region_of_interest(result, vertices_left)

    # RoI Right
    vertices_right = np.array(
        [((int(width*0.9), int(height*0.9)),
          (int(width*0.55), int(height*0.59)),
            (int(width*0.45), int(height*0.59)),
            (int(width*0.65), int(height*0.9)))])
    result_right = helpers.region_of_interest(result, vertices_right)

    # 3 Hough Transform
    result_left = helpers.hough_lines(
        result_left, 1, np.pi/180, 25, 40, 175)  # hier noch interpolieren
    result_right = helpers.hough_lines(
        result_right, 1, np.pi/180, 25, 40, 175)  # hier noch interpolieren
    if(not result_left is None and not result_right is None):
        # 4 GrayScale
        result_left = helpers.grayscale(result_left)
        result_right = helpers.grayscale(result_right)
        gray_left = result_left
        # Finding non zero points
        right_lane = cv2.findNonZero(result_right)
        left_lane = cv2.findNonZero(result_left)
        xr = []
        yr = []
        xl = []
        yl = []
        pts = []
        for pr, pl in zip(right_lane, left_lane):
            xr.append(pr[0][0])
            yr.append(pr[0][1])
            xl.append(pl[0][0])
            yl.append(pl[0][1])

        (ml, bl) = np.polyfit(xl, yl, 1)
        (mr, br) = np.polyfit(xr, yr, 1)
        dy = 10
        for y2 in range(int(height*0.9), int(height*0.65), -dy):
            y1 = y2-dy
            x2r = int((y2-br) / mr)
            x2l = int((y2-bl) / ml)
            x1r = int((y1-br) / mr)
            x1l = int((y1-bl) / ml)
            cv2.line(result, (x1r, y1), (x2r, y2), (250, 255, 255), 5)
            cv2.line(result, (x1l, y1), (x2l, y2), (250, 255, 255), 5)
    else:
        result = last_result
    last_result = result
    # Print Hough Lines on Original Image
    # result = helpers.weighted_img(result, img1, 1, 1.0)

    # Show result
    cv2.imshow("Lane Line Detection", result)
    cv2.imshow("Real Image", gray_left)
    # plt.show()
    # cv2.imshow('frame', result)

    if cv2.waitKey(60) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
