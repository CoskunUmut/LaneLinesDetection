import cv2
import numpy as np
import matplotlib.pyplot as plt
import helpers

"""Test Videos"""
# cap = cv2.VideoCapture("test_videos/solidWhiteRight.mp4")
cap = cv2.VideoCapture("test_videos/challenge.mp4")
#cap = cv2.VideoCapture("test_videos/solidYellowLeft.mp4")
"""Settings"""
# Hyperparameters #
lower_yellow = np.array([10, 75, 175])
up_yellow = np.array([40, 255, 255])
sensivity = 40

lower_white = np.array([0, 0, 255 - sensivity])
up_white = np.array([255, sensivity, 255])

"""Resize Setting"""
_, img1 = cap.read()
height, width, _ = img1.shape
width = int(width / 1.2)
height = int(height / 1.2)

"""Region of Interest"""
vertices = np.array([((int(width * 0.05), int(height * 1)),
                      (int(width * 0.4), int(height * 0.6)),
                      (int(width * 0.6), int(height * 0.6)),
                      (int(width * 0.95), int(height * 1)))])

"""Perspective Transformation"""
# """Hyperparameters"""
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 80
# Set minimum number of pixels found to accept
minpix = 35
# Slide search rectangle
slide_offset = 30
# Set height of windows - based on nwindows above and image shape
window_height = int(height/nwindows)
# Transformation of RoI
src = np.float32(
    [((int(width * 0.05), int(height * 1)),
        (int(width * 0.4), int(height * 0.6)),
        (int(width * 0.6), int(height * 0.6)),
        (int(width * 0.95), int(height * 1)))]
)
dst = np.float32(
    [((int(width * 0), int(height * 1)),
        (int(width * 0), int(height * 0)),
        (int(width * 1), int(height * 0)),
        (int(width * 1), int(height * 1)))]
)
# Transformation calculation
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)


def polyfit_lanes(warped):
    # Rectangle output for debugging
    out_img = warped
    # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Set starting rectangle position for searching lane lines
    cut_warped = warped[int(height-window_height):height][0:width]
    histogram = np.sum(warped[warped.shape[0]//2:, :], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint]) - slide_offset*2
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint + slide_offset*2

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    leftx = []
    lefty = []
    rightx = []
    righty = []
    slide_right = 0
    slide_left = 0

    for w in range(nwindows):
        win_y_low = (height - (w) * window_height)
        win_y_high = (height - (w+1) * window_height)
        win_xleft_low = leftx_base - margin + int(slide_left * slide_offset)
        win_xleft_high = leftx_base + margin + int(slide_left * slide_offset)
        win_xright_low = rightx_base - margin - int(slide_right * slide_offset)
        win_xright_high = rightx_base + margin - \
            int(slide_right * slide_offset)
        good_left_inds = ((nonzeroy <= win_y_low) & (nonzeroy > win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy <= win_y_low) & (nonzeroy > win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        if(len(good_left_inds) > minpix):
            left_lane_inds.append(good_left_inds)
            leftx_base = np.int(
                np.mean(nonzerox[good_left_inds])) + slide_offset
            # cv2.rectangle(out_img, (win_xleft_low, win_y_low),
            #               (win_xleft_high, win_y_high), (255, 0, 0), 3)
            slide_left = 0
        else:
            slide_left = slide_left + 1
        if(len(good_right_inds) > minpix):
            rightx_base = np.int(
                np.mean(nonzerox[good_right_inds])) - slide_offset
            right_lane_inds.append(good_right_inds)
            # cv2.rectangle(out_img, (win_xright_low, win_y_low),
            #               (win_xright_high, win_y_high), (255, 0, 0), 3)
            slide_right = 0
        else:
            slide_right = slide_right+1

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Polyfit x²-Function
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    return ploty, left_fitx, right_fitx


"""Main Loop"""
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, img1 = cap.read()
    if not ret:
        break
    # 1 Resize
    img1 = cv2.resize(img1, (width, height),
                      interpolation=cv2.INTER_AREA)
    result = img1
    # 2 Color Thresholding
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, lower_yellow, up_yellow)
    mask_white = cv2.inRange(hsv, lower_white, up_white)
    mask = mask_white + mask_yellow
    # 3 Prespective Transformation + RoI included
    warped = cv2.warpPerspective(
        mask, M, (width, height), flags=cv2.INTER_LINEAR)
    # 4 Polyfit found lanes
    ploty, left_fitx, right_fitx = polyfit_lanes(warped)
    # 5 Create lane line image to add onto original image
    lane_line = np.zeros([height, width, 3], dtype=np.uint8)
    for i in range(len(ploty)-1):
        cv2.line(lane_line, (int(left_fitx[i]), int(ploty[i])),
                 (int(left_fitx[i+1]), int(ploty[i+1])), (0, 0, 255), 30)
        cv2.line(lane_line, (int(right_fitx[i]), int(ploty[i])),
                 (int(right_fitx[i+1]), int(ploty[i+1])), (0, 0, 255), 30)
    # 6 Inverse Transformation
    rewarped = cv2.warpPerspective(
        lane_line, Minv, (width, height), flags=cv2.INTER_LINEAR)

    # Show Result
    cv2.imshow("Lane Line Detection", cv2.add(img1, rewarped))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        continue


cap.release()
cv2.destroyAllWindows()
