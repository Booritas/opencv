'''
Text skewness correction
This tutorial demonstrates how to correct the skewness in a text.
The program takes as input a skewed source image and shows non skewed text.

Usage:
        python text_skewness_correction.py --image "Image path"
'''

import numpy as np
import cv2 as cv
import sys
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", default="imageTextR.png", help="path to input image file")
    args = vars(parser.parse_args())

    # load the image from disk
    image = ncvslideio.imread(ncvslideio.samples.findFile(args["image"]))
    if image is None:
        print("can't read image " + args["image"])
        sys.exit(-1)
    gray = ncvslideio.cvtColor(image, ncvslideio.COLOR_BGR2GRAY)

    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    thresh = ncvslideio.threshold(gray, 0, 255, ncvslideio.THRESH_BINARY_INV | ncvslideio.THRESH_OTSU)[1]

    # Applying erode filter to remove random noise
    erosion_size = 1
    element = ncvslideio.getStructuringElement(ncvslideio.MORPH_RECT, (2 * erosion_size + 1, 2 * erosion_size + 1), (erosion_size, erosion_size) )
    thresh = ncvslideio.erode(thresh, element)

    coords = ncvslideio.findNonZero(thresh)
    angle = ncvslideio.minAreaRect(coords)[-1]
    # the `ncvslideio.minAreaRect` function returns values in the
    # range [0, 90) if the angle is more than 45 we need to subtract 90 from it
    if angle > 45:
        angle = (angle - 90)

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = ncvslideio.getRotationMatrix2D(center, angle, 1.0)
    rotated = ncvslideio.warpAffine(image, M, (w, h), flags=ncvslideio.INTER_CUBIC, borderMode=ncvslideio.BORDER_REPLICATE)
    ncvslideio.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), ncvslideio.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the output image
    print("[INFO] angle: {:.2f}".format(angle))
    ncvslideio.imshow("Input", image)
    ncvslideio.imshow("Rotated", rotated)
    ncvslideio.waitKey(0)


if __name__ == "__main__":
    print(__doc__)
    main()
    ncvslideio.destroyAllWindows()
