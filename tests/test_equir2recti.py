import os
import cv2
import Equirec2Perspec as E2P

if __name__ == '__main__':
    equ = E2P.Equirectangular('samsung_equirec.png')    # Load equirectangular image

    #
    # FOV unit is degree
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension
    #
    img = equ.GetPerspective(110, -170, -20, 1080, 1920) # Specify parameters(FOV, theta, phi, height, width)

    cv2.imshow("undistorted image", img)
    cv2.waitKey(0)