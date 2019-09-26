import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import helpers
import collections

# images = os.listdir("test_images/")
# Load Image
# path = images[5]
# img1 = cv2.imread("test_images/"+path)
cap = cv2.VideoCapture("test_videos/solidWhiteRight.mp4")
#cap = cv2.VideoCapture("test_videos/challenge.mp4")
#cap = cv2.VideoCapture("test_videos/solidYellowLeft.mp4")


"""Settings"""
lower_yellow = np.array([10, 85, 150])
up_yellow = np.array([40, 255, 255])
sensivity = 30
lower_white = np.array([0, 0, 255-sensivity])
up_white = np.array([255, sensivity, 255])

"""Resize"""
_, img1 = cap.read()
height, width, _ = img1.shape
width = int(width/1.2)
height = int(height/1.2)

"""Region of Interest"""
vertices = np.array(
    [((int(width*0.05), int(height*1)),
        (int(width*0.4), int(height*0.6)),
        (int(width*0.6), int(height*0.6)),
        (int(width*0.95), int(height*1)))])
vertices_left = np.array(
    [((int(width*0.05), int(height*1)),
        (int(width*0.45), int(height*0.6)),
        (int(width*0.475), int(height*0.6)),
        (int(width*0.4), int(height*1)))])
vertices_right = np.array(
    [((int(width*0.95), int(height*1)),
        (int(width*0.55), int(height*0.60)),
        (int(width*0.525), int(height*0.6)),
        (int(width*0.6), int(height*1)))])

"""Main Loop"""
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, img1 = cap.read()
    if not ret:
        break

    img1 = cv2.resize(img1, (width, height),
                      interpolation=cv2.INTER_AREA)
    result = img1
    # 1 Color Thresholding
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, lower_yellow, up_yellow)
    mask_white = cv2.inRange(hsv, lower_white, up_white)
    mask = mask_white+mask_yellow
    # 2 Canny
    result = cv2.Canny(mask, 50, 200)
    canny = result
    # 3 Region of Interest
    result = helpers.region_of_interest(result, vertices)
    left = helpers.region_of_interest(img1, vertices_left)
    right = helpers.region_of_interest(img1, vertices_right)
    roi = result
    # 4 Hough Transform
    try:
        result = helpers.hough_lines(
            result, 1.89, np.pi/180, 25, 25, 125)  # hier noch interpolieren
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        pass

    dy = 15
    dx = 15
    counter = 0
    gray = helpers.grayscale(result)
    xl = []
    yl = []
    xr = []
    yr = []
    for y in range(height-1, int(height*0.6), -dy):
        for x in range(0, int(width/2), dx):
            if(gray[y][x] > 150):
                for ddx in range(x, int(width/2), 5):
                    if(gray[y-dy*2][ddx] > 150):
                        xl.append(ddx)
                        yl.append(y-dy*2)
        for x in range(width-1, int(width/2), -dx):
            if(gray[y][x] > 150):
                for ddx in range(x, int(width/2), -5):
                    if(gray[y-dy*2][ddx] > 150):
                        xr.append(ddx)
                        yr.append(y-dy*2)
    # for i in range(len(pts_left)):
    #     cv2.line(img1, pts_left[i], pts_left2[i], (255, 255, 0), 1)

    # pl = np.polyfit(xl, yl, 2)
    # for x1 in range(int(width*0.1), int(width*0.45), dx):
    #     x2 = x1 + dx
    #     y2 = int(pl[0]*x2*x2+pl[1]*x2+pl[2])
    #     y1 = int(pl[0]*x1*x1+pl[1]*x1+pl[2])
    #     cv2.line(img1, (x2, y2), (x1, y1), (250, 255, 0), 5)
    pr = np.polyfit(xr, yr, 2)
    for x1 in range(int(width*0.9), int(width*0.525), -dx):
        x2 = x1 + dx
        y2 = int(pr[0]*x2*x2+pr[1]*x2+pr[2])
        y1 = int(pr[0]*x1*x1+pr[1]*x1+pr[2])
        cv2.line(img1, (x2, y2), (x1, y1), (250, 255, 0), 5)
        cv2.line(img1, (x2, y2), (x1, y1), (250, 255, 0), 5)

    # for x in range(width, int(width/2), -dx):
    #     print("test2")
    # m, b = np.polyfit(pts_left[0], pts_left[1], 1)
    # for y in range(height-1, int(height/2), -dy):

    #         if(gray[y][x] > 150):

    # Put Hough Lines on original image
    # result = helpers.weighted_img(result, img1, 1, 1.0)

    # Show Result
    cv2.imshow("Lane Line Detection", img1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
