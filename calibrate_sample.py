import numpy as np
import cv2
import glob

# Define the checkerboard dimensions
checkerboard_size = (10, 7)
# Prepare object points (0,0,0), (1,0,0), (2,0,0), ..., (9,6,0)
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all images
object_points = []  # 3d points in real world space
image_points = []  # 2d points in image plane

# Load all images
images = glob.glob('./sample/*.jpeg')

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Failed to load image {fname}")
        continue

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve preprocessing
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)

    # Find initial corners
    ret, corners = cv2.findChessboardCorners(img_gray, checkerboard_size, 
                                             flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                   cv2.CALIB_CB_FAST_CHECK + 
                                                   cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret:
        print(f"Corners found in {fname}")

        # Define the criteria for corner refinement
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)

        object_points.append(objp)
        image_points.append(corners)

        # Draw and display corners
        fnl = cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        #cv2.imshow("Final", fnl)
        #cv2.waitKey(0)
    else:
        print(f"No Checkerboard Found in {fname}")

cv2.destroyAllWindows()

# Camera calibration
if object_points and image_points:
    img_size = (img.shape[1], img.shape[0])
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_size, None, None)

    # Print the calibration results in a readable format
    print("\n=== Camera Calibration Results ===")
    print(f"Reprojection Error: {retval}\n")

    print("Camera Matrix:")
    print(cameraMatrix, "\n")

    print("Distortion Coefficients:")
    print(distCoeffs.ravel(), "\n")

    print("Rotation Vectors:")
    for i, rvec in enumerate(rvecs):
        print(f"Image {i + 1}:")
        print(rvec.ravel(), "\n")

    print("Translation Vectors:")
    for i, tvec in enumerate(tvecs):
        print(f"Image {i + 1}:")
        print(tvec.ravel(), "\n")
else:
    print("Not enough points for calibration")