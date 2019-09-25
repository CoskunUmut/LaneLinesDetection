import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import helpers


# images = os.listdir("test_images/")
# Load Image
# path = images[5]
# img1 = cv2.imread("test_images/"+path)
# cap = cv2.VideoCapture("test_videos/solidWhiteRight.mp4")
#cap = cv2.VideoCapture("test_videos/challenge.mp4")
cap = cv2.VideoCapture("test_videos/solidYellowLeft.mp4")
_, img1 = cap.read()
height, width, _ = img1.shape


while(cap.isOpened()):
    # Capture frame-by-frame
    ret, img1 = cap.read()
    result = img1

    # # 1.1 Thresholding
    # _, result = cv2.threshold(result, 215, 255, cv2.THRESH_BINARY)
    # # 1.2 Blurring for smoother edges
    # result = helpers.gaussian_blur(result, 9)
    # # # 2 Canny
    # canny1 = cv2.Canny(result, 75, 200)
    # result = canny1

    # RoI Left
    vertices = np.array(
        [((int(width*0.1), int(height*0.9)),
          (int(width*0.4), int(height*0.58)),
            (int(width*0.55), int(height*0.58)),
            (int(width*0.35), int(height*0.9)))])

    # RoI Right
    vertices2 = np.array(
        [((int(width*0.9), int(height*0.9)),
          (int(width*0.6), int(height*0.58)),
            (int(width*0.5), int(height*0.58)),
            (int(width*0.65), int(height*0.9)))])
    result = helpers.region_of_interest(result, vertices, vertices2)
    # # 3 Hough Transform
    # try:
    #     result = helpers.hough_lines(
    #         result, 1, np.pi/180, 50, 25, 75)  # hier noch interpolieren
    #     result = helpers.weighted_img(result, img1, 1, 1.0)
    # except:
    #     _, result = cv2.threshold(img1, 175, 215, cv2.THRESH_BINARY)
    #     result = helpers.gaussian_blur(result, 9)
    #     result = cv2.Canny(result, 75, 200)
    #     result = helpers.region_of_interest(result, vertices)
    #     result = helpers.hough_lines(
    #         result, 1, np.pi/180, 50, 25, 75)  # hier noch interpolieren
    #     result = helpers.weighted_img(result, img1, 1, 1.0)

    # Show result
    cv2.imshow("Lane Line Detection", result)
    # cv2.imshow("Real Image", img1)
    # plt.show()
    # cv2.imshow('frame', result)
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
