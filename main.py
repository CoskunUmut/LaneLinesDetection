import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import helpers

#images = os.listdir("test_images/")

# Load Image
# path = images[5]
# img1 = cv2.imread("test_images/"+path)

cap = cv2.VideoCapture("test_videos/solidWhiteRight.mp4")

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, img1 = cap.read()
    # 1 Grayscale Image
    gray1 = helpers.grayscale(img1)
    result = gray1

    # 2 Canny
    canny1 = helpers.canny(result, 200, 350)
    result = canny1

    # 3 Region of Interest
    vertices = np.array([((50, 540), (440, 350), (540, 350), (950, 540))])
    roi1 = helpers.region_of_interest(result, vertices)
    result = roi1

    # 4 Hough Transformation
    hough1 = helpers.hough_lines(result, 1, np.pi / 180, 25, 50, 200)
    result = hough1
    hough1_gray = helpers.grayscale(hough1)
    hough1_gray = np.swapaxes(hough1_gray, 0, 1)

    # # Interpolating left/right lane lines
    # counter = 0
    # lines_right = []
    # lines_left = []
    # sx = 10
    # sy = 10
    # stepx = 50

    # # Right Lane
    # for y in range(round(hough1_gray.shape[1]/sy)-1):
    #     for x in range(round(hough1_gray.shape[0]/sx)):
    #         if(hough1_gray[sx*x][sy*y] > 0):
    #             dx = sx*x+stepx
    #             if(dx < hough1_gray.shape[0]):
    #                 for dy in range(sy*y, 1000):
    #                     if(dy < hough1_gray.shape[1]):
    #                         if(hough1_gray[dx][dy] > 0):
    #                             lines_right.append(
    #                                 [sx*x, sy*y, dx, dy])
    #                             counter += 1

    # x1, y1, _, _ = lines_right[0]
    # x2, y2, _, _ = lines_right[len(lines_right)-1]
    # m = ((y2-y1) / (x2-x1))
    # b = int(y1 + (-m * x1))
    # y3 = hough1_gray.shape[1]
    # x3 = round((y3 - b)/m)
    # lines_right = []
    # lines_right.append([x1, y1, x3, y3])

    # counter = 0
    # for y in range(round(hough1_gray.shape[1]/sy)-1):
    #     for x in range(round(hough1_gray.shape[0]/sx)):
    #         if(hough1_gray[sx*x][sy*y] > 0):
    #             dx = sx*x-stepx
    #             if(dx > 0):
    #                 for dy in range(sy*y, 1000):
    #                     if(dy < hough1_gray.shape[1]):
    #                         if(hough1_gray[dx][dy] > 0):
    #                             lines_left.append(
    #                                 [sx*x, sy*y, dx, dy])
    #                             counter += 1
    # x1, y1, _, _ = lines_left[0]
    # x2, y2, _, _ = lines_left[len(lines_left)-1]
    # m = ((y2-y1) / (x2-x1))
    # b = int(y1 + (-m * x1))
    # y3 = hough1_gray.shape[1]
    # x3 = round((y3 - b)/m)
    # lines_left = []
    # lines_left.append([x1, y1, x3, y3])

    # result = img1
    # helpers.draw_lines2(result, lines_right)
    # helpers.draw_lines2(result, lines_left)

    # Add result to original image
    result = helpers.weighted_img(hough1, img1, 1.0, 1.0)

    # Show result
    # cv2.imshow("canny", canny1)
    # cv2.imshow("glbur", gblur1)
    # cv2.imshow("roi", roi1)
    #cv2.imshow("hough", hough1)
    # cv2.imshow("Lane Line Detection", result)
    plt.imshow(result)
    plt.show()
    #cv2.imshow('frame', result)
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
