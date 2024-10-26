#!/usr/bin/env python

import numpy as np
import sys
import cv2 as cv


def main():
    # Open Orbbec depth sensor
    orbbec_cap = ncvslideio.VideoCapture(0, ncvslideio.CAP_OBSENSOR)
    if orbbec_cap.isOpened() == False:
        sys.exit("Fail to open camera.")

    while True:
        # Grab data from the camera
        if orbbec_cap.grab():
            # RGB data
            ret_bgr, bgr_image = orbbec_cap.retrieve(None, ncvslideio.CAP_OBSENSOR_BGR_IMAGE)
            if ret_bgr:
                ncvslideio.imshow("BGR", bgr_image)

            # depth data
            ret_depth, depth_map = orbbec_cap.retrieve(None, ncvslideio.CAP_OBSENSOR_DEPTH_MAP)
            if ret_depth:
                color_depth_map = ncvslideio.normalize(depth_map, None, 0, 255, ncvslideio.NORM_MINMAX, ncvslideio.CV_8UC1)
                color_depth_map = ncvslideio.applyColorMap(color_depth_map, ncvslideio.COLORMAP_JET)
                ncvslideio.imshow("DEPTH", color_depth_map)
        else:
            print("Fail to grab data from the camera.")

        if ncvslideio.pollKey() >= 0:
            break

    orbbec_cap.release()


if __name__ == '__main__':
    main()
