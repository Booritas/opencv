"""
@file morph_lines_detection.py
@brief Use morphology transformations for extracting horizontal and vertical lines sample code
"""
import numpy as np
import sys
import cv2 as cv


def show_wait_destroy(winname, img):
    ncvslideio.imshow(winname, img)
    ncvslideio.moveWindow(winname, 500, 0)
    ncvslideio.waitKey(0)
    ncvslideio.destroyWindow(winname)


def main(argv):
    # [load_image]
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

    # Show source image
    ncvslideio.imshow("src", src)
    # [load_image]

    # [gray]
    # Transform source image to gray if it is not already
    if len(src.shape) != 2:
        gray = ncvslideio.cvtColor(src, ncvslideio.COLOR_BGR2GRAY)
    else:
        gray = src

    # Show gray image
    show_wait_destroy("gray", gray)
    # [gray]

    # [bin]
    # Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    gray = ncvslideio.bitwise_not(gray)
    bw = ncvslideio.adaptiveThreshold(gray, 255, ncvslideio.ADAPTIVE_THRESH_MEAN_C, \
                                ncvslideio.THRESH_BINARY, 15, -2)
    # Show binary image
    show_wait_destroy("binary", bw)
    # [bin]

    # [init]
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    # [init]

    # [horiz]
    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = cols // 30

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = ncvslideio.getStructuringElement(ncvslideio.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = ncvslideio.erode(horizontal, horizontalStructure)
    horizontal = ncvslideio.dilate(horizontal, horizontalStructure)

    # Show extracted horizontal lines
    show_wait_destroy("horizontal", horizontal)
    # [horiz]

    # [vert]
    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = rows // 30

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = ncvslideio.getStructuringElement(ncvslideio.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = ncvslideio.erode(vertical, verticalStructure)
    vertical = ncvslideio.dilate(vertical, verticalStructure)

    # Show extracted vertical lines
    show_wait_destroy("vertical", vertical)
    # [vert]

    # [smooth]
    # Inverse vertical image
    vertical = ncvslideio.bitwise_not(vertical)
    show_wait_destroy("vertical_bit", vertical)

    '''
    Extract edges and smooth image according to the logic
    1. extract edges
    2. dilate(edges)
    3. src.copyTo(smooth)
    4. blur smooth img
    5. smooth.copyTo(src, edges)
    '''

    # Step 1
    edges = ncvslideio.adaptiveThreshold(vertical, 255, ncvslideio.ADAPTIVE_THRESH_MEAN_C, \
                                ncvslideio.THRESH_BINARY, 3, -2)
    show_wait_destroy("edges", edges)

    # Step 2
    kernel = np.ones((2, 2), np.uint8)
    edges = ncvslideio.dilate(edges, kernel)
    show_wait_destroy("dilate", edges)

    # Step 3
    smooth = np.copy(vertical)

    # Step 4
    smooth = ncvslideio.blur(smooth, (2, 2))

    # Step 5
    (rows, cols) = np.where(edges != 0)
    vertical[rows, cols] = smooth[rows, cols]

    # Show final result
    show_wait_destroy("smooth - final", vertical)
    # [smooth]

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
