"""
@file sobel_demo.py
@brief Sample code using Sobel and/or Scharr OpenCV functions to make a simple Edge Detector
"""
import sys
import cv2 as cv


def main(argv):
    ## [variables]
    # First we declare the variables we are going to use
    window_name = ('Sobel Demo - Simple Edge Detector')
    scale = 1
    delta = 0
    ddepth = ncvslideio.CV_16S
    ## [variables]

    ## [load]
    # As usual we load our source image (src)
    # Check number of arguments
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1

    # Load the image
    src = ncvslideio.imread(argv[0], ncvslideio.IMREAD_COLOR)

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1
    ## [load]

    ## [reduce_noise]
    # Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    src = ncvslideio.GaussianBlur(src, (3, 3), 0)
    ## [reduce_noise]

    ## [convert_to_gray]
    # Convert the image to grayscale
    gray = ncvslideio.cvtColor(src, ncvslideio.COLOR_BGR2GRAY)
    ## [convert_to_gray]

    ## [sobel]
    # Gradient-X
    # grad_x = ncvslideio.Scharr(gray,ddepth,1,0)
    grad_x = ncvslideio.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=ncvslideio.BORDER_DEFAULT)

    # Gradient-Y
    # grad_y = ncvslideio.Scharr(gray,ddepth,0,1)
    grad_y = ncvslideio.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=ncvslideio.BORDER_DEFAULT)
    ## [sobel]

    ## [convert]
    # converting back to uint8
    abs_grad_x = ncvslideio.convertScaleAbs(grad_x)
    abs_grad_y = ncvslideio.convertScaleAbs(grad_y)
    ## [convert]

    ## [blend]
    ## Total Gradient (approximate)
    grad = ncvslideio.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    ## [blend]

    ## [display]
    ncvslideio.imshow(window_name, grad)
    ncvslideio.waitKey(0)
    ## [display]

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
