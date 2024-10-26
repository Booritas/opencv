from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

src = None
erosion_size = 0
max_elem = 2
max_kernel_size = 21
title_trackbar_element_shape = 'Element:\n 0: Rect \n 1: Cross \n 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
title_erosion_window = 'Erosion Demo'
title_dilation_window = 'Dilation Demo'


## [main]
def main(image):
    global src
    src = ncvslideio.imread(ncvslideio.samples.findFile(image))
    if src is None:
        print('Could not open or find the image: ', image)
        exit(0)

    ncvslideio.namedWindow(title_erosion_window)
    ncvslideio.createTrackbar(title_trackbar_element_shape, title_erosion_window, 0, max_elem, erosion)
    ncvslideio.createTrackbar(title_trackbar_kernel_size, title_erosion_window, 0, max_kernel_size, erosion)

    ncvslideio.namedWindow(title_dilation_window)
    ncvslideio.createTrackbar(title_trackbar_element_shape, title_dilation_window, 0, max_elem, dilatation)
    ncvslideio.createTrackbar(title_trackbar_kernel_size, title_dilation_window, 0, max_kernel_size, dilatation)

    erosion(0)
    dilatation(0)
    ncvslideio.waitKey()
## [main]

# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return ncvslideio.MORPH_RECT
    elif val == 1:
        return ncvslideio.MORPH_CROSS
    elif val == 2:
        return ncvslideio.MORPH_ELLIPSE


## [erosion]
def erosion(val):
    erosion_size = ncvslideio.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window)
    erosion_shape = morph_shape(ncvslideio.getTrackbarPos(title_trackbar_element_shape, title_erosion_window))

    ## [kernel]
    element = ncvslideio.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    ## [kernel]
    erosion_dst = ncvslideio.erode(src, element)
    ncvslideio.imshow(title_erosion_window, erosion_dst)
## [erosion]


## [dilation]
def dilatation(val):
    dilatation_size = ncvslideio.getTrackbarPos(title_trackbar_kernel_size, title_dilation_window)
    dilation_shape = morph_shape(ncvslideio.getTrackbarPos(title_trackbar_element_shape, title_dilation_window))

    element = ncvslideio.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    dilatation_dst = ncvslideio.dilate(src, element)
    ncvslideio.imshow(title_dilation_window, dilatation_dst)
## [dilation]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for Eroding and Dilating tutorial.')
    parser.add_argument('--input', help='Path to input image.', default='LinuxLogo.jpg')
    args = parser.parse_args()

    main(args.input)
