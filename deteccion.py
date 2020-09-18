import numpy as np # just for matrix manipulation, C/C++ use cv::Mat
import cv2
# find contours.
frame = cv2.imread('prueba.jpeg',0)

cv2.line(frame, (650, 0), (650, 1350), (255, 0, 0), 4)
contours,h = cv2.findContours( frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE )
blank = np.zeros( frame.shape[0:2] )
cv2.imshow(frame)
img1 = cv2.drawContours( blank.copy(), contours, 0, 1 )
img2 = cv2.drawContours( blank.copy(), contours, 1, 1 )

intersection = np.logical_and( img1, img2 )

# OR we could just add img1 to img2 and pick all points that sum to 2 (1+1=2):
#intersection2 = (img1+img2)==2