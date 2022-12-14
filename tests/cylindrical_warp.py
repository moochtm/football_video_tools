import cv2
import numpy as np


def cv_cylindricalWarp(img):
    pass


def cylindricalWarp(img, K=None):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h_, w_ = img.shape[:2]

    if K is None:
        K = np.array([[800, 0, w_ / 2], [0, 800, h_ / 2], [0, 0, 1]])  # mock intrinsics

    # pixel coordinates
    y_i, x_i = np.indices((h_, w_))
    X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)  # to homog
    Kinv = np.linalg.inv(K)
    X = Kinv.dot(X.T).T  # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(
        w_ * h_, 3
    )
    B = K.dot(A.T).T  # project back to image-pixels plane
    # back from homog coords
    B = B[:, :-1] / B[:, [-1]]
    # make sure warp coords only within image bounds
    B[(B[:, 0] < 0) | (B[:, 0] >= w_) | (B[:, 1] < 0) | (B[:, 1] >= h_)] = -1
    B = B.reshape(h_, w_, -1)

    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # for transparent borders...
    # warp the image according to cylindrical coords
    return cv2.remap(
        img_rgba,
        B[:, :, 0].astype(np.float32),
        B[:, :, 1].astype(np.float32),
        cv2.INTER_AREA,
        borderMode=cv2.BORDER_TRANSPARENT,
    )


if __name__ == "__main__":
    img = cv2.imread("/Users/home/Pictures/vlcsnap-2022-10-23-19h16m37s493.png")
    h, w = img.shape[:2]
    K = np.array([[800, 0, w / 2], [0, 800, h / 2], [0, 0, 1]])  # mock intrinsics
    img_cyl = cylindricalWarp(img, K)
    cv2.imwrite("/Users/home/Pictures/image_cyl_R.png", img_cyl)
