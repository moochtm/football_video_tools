import cv2
import numpy as np
import glob
import pickle
import os


def correct_distortion(image, config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError

    filehandler = open(config_file, 'rb')
    ret, mtx, dist, rvecs, tvecs = pickle.load(filehandler)
    filehandler.close()

    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)

    h, w = image.shape[:2]

    # Refining the camera matrix using parameters obtained by calibration
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    return cv2.undistort(image, mtx, dist, None, new_camera_mtx)


if __name__ == '__main__':

    checkerboard_images_path = '/Users/home/Code/football_video_tools/tests/UnderstandingLensDistortion/output1 33.7Mbps/*.png'
    config_file_name = 'input/blah.pickle'

    # Defining the dimensions of checkerboard
    CHECKERBOARD = (7, 10)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []

    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = []

    # Defining the world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of individual image stored in a given directory
    # images = glob.glob('./output1 33.7Mbps/*.png')
    images = glob.glob(checkerboard_images_path)
    for fname in images:
        print(fname)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        # If desired number of corners are found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                 cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        """
        If desired number of corner are detected,
        we refine the pixel coordinates and display 
        them on the images of checker board
        """
        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        # cv2.imshow('img', img)
        # cv2.waitKey(0)

    # cv2.destroyAllWindows()

    # h, w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    camera_calib = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    filehandler = open(config_file_name, 'wb')
    pickle.dump(camera_calib, filehandler)
    filehandler.close()
