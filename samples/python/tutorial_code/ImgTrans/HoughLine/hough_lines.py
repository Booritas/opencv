"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 as cv
import numpy as np


def main(argv):
    ## [load]
    default_file = 'sudoku.png'
    filename = argv[0] if len(argv) > 0 else default_file

    # Loads an image
    src = ncvslideio.imread(ncvslideio.samples.findFile(filename), ncvslideio.IMREAD_GRAYSCALE)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    ## [load]

    ## [edge_detection]
    # Edge detection
    dst = ncvslideio.Canny(src, 50, 200, None, 3)
    ## [edge_detection]

    # Copy edges to the images that will display the results in BGR
    cdst = ncvslideio.cvtColor(dst, ncvslideio.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    ## [hough_lines]
    #  Standard Hough Line Transform
    lines = ncvslideio.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
    ## [hough_lines]
    ## [draw_lines]
    # Draw the lines
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

            ncvslideio.line(cdst, pt1, pt2, (0,0,255), 3, ncvslideio.LINE_AA)
    ## [draw_lines]

    ## [hough_lines_p]
    # Probabilistic Line Transform
    linesP = ncvslideio.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    ## [hough_lines_p]
    ## [draw_lines_p]
    # Draw the lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            ncvslideio.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, ncvslideio.LINE_AA)
    ## [draw_lines_p]
    ## [imshow]
    # Show results
    ncvslideio.imshow("Source", src)
    ncvslideio.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    ncvslideio.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    ## [imshow]
    ## [exit]
    # Wait and Exit
    ncvslideio.waitKey()
    return 0
    ## [exit]

if __name__ == "__main__":
    main(sys.argv[1:])
