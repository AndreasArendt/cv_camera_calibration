import numpy as np
import cv2
import glob

# Define the checkerboard dimensions
checkerboard_size = (10, 7)

# Define the size of a square in your defined unit (e.g., cm or mm)
square_size = 2.5  # example size in cm

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (9,6,0)
# Multiply by the actual square size to scale to real-world coordinates
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

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

    # Find the chessboard corners
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
        cv2.imshow("Final", fnl)
        cv2.waitKey(0)
    else:
        print(f"No Checkerboard Found in {fname}")

cv2.destroyAllWindows()

# Camera calibration
if object_points and image_points:
    img_size = (img.shape[1], img.shape[0])
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_size, None, None)

    # Print the calibration results in a readable format
    print("\n=== Camera Calibration Results ===")
    print(f"Reprojection Error: {retval:.4f}\n")

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

    # Compute and display per-image reprojection errors
    total_error = 0
    errors = []
    for i in range(len(object_points)):
        img_points2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
        error = cv2.norm(image_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
        total_error += error
        errors.append(error)
        print(f"Image {i + 1} Reprojection Error: {error:.4f}")

    mean_error = total_error / len(object_points)
    print(f"\nMean Reprojection Error: {mean_error:.4f}")

else:
    print("Not enough points for calibration")
