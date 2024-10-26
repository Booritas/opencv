from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse

source_window = 'Source image'
corners_window = 'Corners detected'
max_thresh = 255

def cornerHarris_demo(val):
    thresh = val

    # Detector parameters
    blockSize = 2
    apertureSize = 3
    k = 0.04

    # Detecting corners
    dst = ncvslideio.cornerHarris(src_gray, blockSize, apertureSize, k)

    # Normalizing
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    ncvslideio.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=ncvslideio.NORM_MINMAX)
    dst_norm_scaled = ncvslideio.convertScaleAbs(dst_norm)

    # Drawing a circle around corners
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i,j]) > thresh:
                ncvslideio.circle(dst_norm_scaled, (j,i), 5, (0), 2)

    # Showing the result
    ncvslideio.namedWindow(corners_window)
    ncvslideio.imshow(corners_window, dst_norm_scaled)

# Load source image and convert it to gray
parser = argparse.ArgumentParser(description='Code for Harris corner detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='building.jpg')
args = parser.parse_args()

src = ncvslideio.imread(ncvslideio.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

src_gray = ncvslideio.cvtColor(src, ncvslideio.COLOR_BGR2GRAY)

# Create a window and a trackbar
cv.namedWindow(source_window)
thresh = 200 # initial threshold
cv.createTrackbar('Threshold: ', source_window, thresh, max_thresh, cornerHarris_demo)
cv.imshow(source_window, src)
cornerHarris_demo(thresh)

cv.waitKey()
