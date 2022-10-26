import numpy as np
import cv2
import cylindrical_warp
import test_recanvas

# def rotate_image(image, angle):
#     image_center = tuple(np.array(image.shape[1::-1]) / 2)
#     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#     print(image.shape[1])
#     print(image.shape[1::-1])
#     # result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
#     result = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[1]), flags=cv2.INTER_LINEAR)
#     return result


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2,
    )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


if __name__ == "__main__":
    img = cv2.imread("/Users/home/Pictures/vlcsnap-2022-10-23-19h16m08s292.png")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    img = rotate_image(img, 11)
    h_, w_ = img.shape[:2]

    cv2.imshow("output", img)
    cv2.waitKey(0)

    img = test_recanvas.pad(img, top=h_ * 0.1)
    cv2.imshow("output", img)
    cv2.waitKey(0)

    h_, w_ = img.shape[:2]
    K = np.array(
        [[w_ * 0.6, 0, w_ / 2], [0, h_ * 0.6, h_ / 2], [0, 0, 1]]
    )  # mock intrinsics
    img = cylindrical_warp.cylindricalWarp(img, K)
    # cv2.imwrite("/Users/home/Pictures/image_cyl_R.png", img_cyl)
    cv2.imshow("output", img)
    cv2.waitKey(0)
    cv2.imwrite("/Users/home/Pictures/vlcsnap-2022-10-23-19h16m08s292_blah.png", img)
