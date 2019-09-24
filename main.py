import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import helpers


def average_brightness(roi):
    print(roi.shape)


# images = os.listdir("test_images/")

# Load Image
# path = images[5]
# img1 = cv2.imread("test_images/"+path)

#cap = cv2.VideoCapture("test_videos/solidWhiteRight.mp4")
cap = cv2.VideoCapture("test_videos/challenge.mp4")
#cap = cv2.VideoCapture("test_videos/solidYellowLeft.mp4")
_, img1 = cap.read()
height, width, _ = img1.shape


while(cap.isOpened()):
    # Capture frame-by-frame
    ret, img1 = cap.read()
    result = img1
    # 1 Grayscale Image
    # gray1 = helpers.grayscale(result)
    # result = gray1
    # 1.2
    # gauss1 = helpers.gaussian_blur(result, 1)
    # result = gauss1
    # 1.1 Thresholding
    # average brightness for dynamic threshold
    _, result = cv2.threshold(result, 215, 255, cv2.THRESH_BINARY)
    result = helpers.gaussian_blur(result, 9)

    # _, thresh2 = cv2.threshold(gray1, 100, 255, cv2.THRESH_BINARY_INV)
    # _, thresh3 = cv2.threshold(gray1, 225, d255, cv2.THRESH_TRUNC)
    # _, thresh4 = cv2.threshold(gray1, 175, 255, cv2.THRESH_TOZERO)

    # 2 Canny
    canny1 = cv2.Canny(result, 75, 200)
    result = canny1

    # RoI
    # vertices = np.array(
    #     [((int(width*0.05), int(height*0.9)),
    #       (int(width*0.4), int(height*0.65)),
    #         (int(width*0.6), int(height*0.65)),
    #         (int(width*0.95), int(height*0.9)))])
    vertices = np.array(
        [((int(width*0.15), int(height*0.9)),
          (int(width*0.475), int(height*0.6)),
            (int(width*0.55), int(height*0.6)),
            (int(width*0.35), int(height*0.9)))])
    result = helpers.region_of_interest(result, vertices)
    average_brightness(result)
    # 3 Hough Transform
    try:
        result = helpers.hough_lines(
            result, 1, np.pi/180, 50, 25, 75)  # hier noch interpolieren
        result = helpers.weighted_img(result, img1, 1, 1.0)
    except:
        _, result = cv2.threshold(img1, 175, 215, cv2.THRESH_BINARY)
        result = helpers.gaussian_blur(result, 9)
        result = cv2.Canny(result, 75, 200)
        result = helpers.region_of_interest(result, vertices)
        result = helpers.hough_lines(
            result, 1, np.pi/180, 50, 25, 75)  # hier noch interpolieren
        result = helpers.weighted_img(result, img1, 1, 1.0)

    # print(len(result))
    # 4 RoI
    # result[x/2,y=max-10%]
   # hough1 = helpers.hough_lines(result, 1, np.pi / 180, 25, 50, 200)
    # result = hough1
    # 3 Region of Interest
    # vertices = np.array([((50, 540), (440, 350), (540, 350), (950, 540))])
    # roi1 = helpers.region_of_interest(result, vertices)
    # result = roi1

    # # 4 Hough Transformation
    # hough1 = helpers.hough_lines(result, 1, np.pi / 180, 25, 50, 200)
    # result = hough1
    # hough1_gray = helpers.grayscale(hough1)
    # hough1_gray = np.swapaxes(hough1_gray, 0, 1)

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
    # result = helpers.weighted_img(hough1, img1, 1.0, 1.0)

    # Show result
    # cv2.imshow("canny", canny1)
    # cv2.imshow("glbur", gblur1)
    # cv2.imshow("roi", roi1)
    # cv2.imshow("hough", hough1)
    cv2.imshow("Lane Line Detection", result)
    #cv2.imshow("Real Image", img1)
    # plt.imshow(result)
    # plt.show()
    # cv2.imshow('frame', result)
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
