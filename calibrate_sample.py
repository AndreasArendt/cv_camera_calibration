import numpy as np
import cv2
import glob
  
images = glob.glob('./sample/*.jpeg')
 
for fname in images:
    img = cv2.imread(fname)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve preprocessing
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)

    ret, corners = cv2.findChessboardCorners(img_gray, (10, 7), 
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                   cv2.CALIB_CB_FAST_CHECK + 
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    if ret:
        print(corners)
        fnl = cv2.drawChessboardCorners(img, (10, 7), corners, ret)
        cv2.imshow("fnl", fnl)
        cv2.waitKey(0)
    else:
        print("No Checkerboard Found")