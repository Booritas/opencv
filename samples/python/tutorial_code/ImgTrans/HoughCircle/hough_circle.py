import sys
import cv2 as cv
import numpy as np


def main(argv):
    ## [load]
    default_file = 'smarties.png'
    filename = argv[0] if len(argv) > 0 else default_file

    # Loads an image
    src = ncvslideio.imread(ncvslideio.samples.findFile(filename), ncvslideio.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_circle.py [image_name -- default ' + default_file + '] \n')
        return -1
    ## [load]

    ## [convert_to_gray]
    # Convert it to gray
    gray = ncvslideio.cvtColor(src, ncvslideio.COLOR_BGR2GRAY)
    ## [convert_to_gray]

    ## [reduce_noise]
    # Reduce the noise to avoid false circle detection
    gray = ncvslideio.medianBlur(gray, 5)
    ## [reduce_noise]

    ## [houghcircles]
    rows = gray.shape[0]
    circles = ncvslideio.HoughCircles(gray, ncvslideio.HOUGH_GRADIENT, 1, rows / 8,
                               param1=100, param2=30,
                               minRadius=1, maxRadius=30)
    ## [houghcircles]

    ## [draw]
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            ncvslideio.circle(src, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            ncvslideio.circle(src, center, radius, (255, 0, 255), 3)
    ## [draw]

    ## [display]
    ncvslideio.imshow("detected circles", src)
    ncvslideio.waitKey(0)
    ## [display]

    return 0


if __name__ == "__main__":
    main(sys.argv[1:])
