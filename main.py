#!/usr/bin/env python

import cv2
import numpy as np
import os
import glob
import time

# Defining the dimensions of checkerboard
CHECKERBOARD = (10, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob("./images/waveshare-160-noir/*.png")
images = images[:]
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(
        img, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    # ret, corners = cv2.findChessboardCorners(img, CHECKERBOARD, cv2.CALIB_CB_FAST_CHECK)
    # print("ret+corners: ", ret, corners)
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:  # noqa: E712
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

    # cv2.imshow("img", img)
    # cv2.waitKey(0)

cv2.destroyAllWindows()

h, w = img.shape[:2]

N_imm = len(images)  # number of calibration images
K = np.zeros((3, 3))
D = np.zeros((4, 1))
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]

# K: the new camera matrix
# D: distortion coefficients
reproj_error, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
)

map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)


def calculate_fov_from_corners(*, camera_matrix, dist_coeffs, image_width, image_height):
    # Define corner points of the image
    corners = np.array(
        [
            [[0.0, image_height / 2]],  # Left edge, center
            [[image_width, image_height / 2]],  # Right edge, center
            [[image_width / 2, 0.0]],  # Top edge, center
            [[image_width / 2, image_height]],  # Bottom edge, center
        ],
        dtype=np.float32,
    )

    # Undistort the corner points
    undistorted_corners = cv2.fisheye.undistortPoints(corners, camera_matrix, dist_coeffs)

    # Calculate angles from center
    angles = []
    for point in undistorted_corners:
        x, y = point[0]
        angle = np.arctan(np.sqrt(x * x + y * y))  # Angle from optical axis
        angles.append(np.degrees(angle))

    horizontal_fov = 2 * angles[0]  # Left edge angle * 2
    vertical_fov = 2 * angles[2]  # Top edge angle * 2

    return horizontal_fov, vertical_fov


fov_x, fov_y = calculate_fov_from_corners(camera_matrix=K, dist_coeffs=D, image_width=w, image_height=h)
print(f"FoV (from corners): ({fov_x}°, {fov_y}°)")


def calculate_angle_for(*, point: tuple[int, int]):
    distorted_pt_px = np.array([[point]], dtype=np.float32)  # Must be in a nested array
    undistorted_pt_norm = cv2.fisheye.undistortPoints(distorted_pt_px, K, D)
    x_norm = undistorted_pt_norm[0][0][0]

    # Now, your code is perfect
    angle_rad = np.arctan(x_norm)
    angle_deg = np.degrees(angle_rad)

    print(f"Original distorted pixel: {distorted_pt_px[0][0]}")
    print(f"Normalized coordinate (x_norm): {x_norm:.4f}")
    print(f"Angle from center: {angle_deg:.2f} degrees")


# # without cropping
img_dim = img.shape[:2][::-1]
DIM = img_dim  # dimension of the images used for calibration
scaled_K = K * img_dim[0] / DIM[0]
scaled_K[2][2] = 1.0
balance = 1
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, img_dim, np.eye(3), balance=balance)
u_map1, u_map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, img_dim, cv2.CV_16SC2)

for fname in images:
    img = cv2.imread(fname)

    t1 = time.perf_counter()
    # Method 1 to undistort the image
    # remap with cropping
    img_undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    time_elapsed = time.perf_counter() - t1
    print("Undistortion took: ", round(time_elapsed * 1000, 3), "ms.")

    # cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # numpy_horizontal = np.hstack((img, img_undistorted))

    # without cropping
    undist_image2 = cv2.remap(img, u_map1, u_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    numpy_horizontal = np.hstack((img, img_undistorted, undist_image2))

    cv2.imshow("Numpy Horizontal", numpy_horizontal)

    distorted_point = (526, 0)
    calculate_angle_for(point=distorted_point)

    cv2.waitKey(0)


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

print("total error: ", mean_error / len(objpoints))
