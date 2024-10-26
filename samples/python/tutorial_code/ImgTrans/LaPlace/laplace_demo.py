"""
@file laplace_demo.py
@brief Sample code showing how to detect edges using the Laplace operator
"""
import sys
import cv2 as cv

def main(argv):
    # [variables]
    # Declare the variables we are going to use
    ddepth = ncvslideio.CV_16S
    kernel_size = 3
    window_name = "Laplace Demo"
    # [variables]

    # [load]
    imageName = argv[0] if len(argv) > 0 else 'lena.jpg'

    src = ncvslideio.imread(ncvslideio.samples.findFile(imageName), ncvslideio.IMREAD_COLOR) # Load an image

    # Check if image is loaded fine
    if src is None:
        print ('Error opening image')
        print ('Program Arguments: [image_name -- default lena.jpg]')
        return -1
    # [load]

    # [reduce_noise]
    # Remove noise by blurring with a Gaussian filter
    src = ncvslideio.GaussianBlur(src, (3, 3), 0)
    # [reduce_noise]

    # [convert_to_gray]
    # Convert the image to grayscale
    src_gray = ncvslideio.cvtColor(src, ncvslideio.COLOR_BGR2GRAY)
    # [convert_to_gray]

    # Create Window
    ncvslideio.namedWindow(window_name, ncvslideio.WINDOW_AUTOSIZE)

    # [laplacian]
    # Apply Laplace function
    dst = ncvslideio.Laplacian(src_gray, ddepth, ksize=kernel_size)
    # [laplacian]

    # [convert]
    # converting back to uint8
    abs_dst = ncvslideio.convertScaleAbs(dst)
    # [convert]

    # [display]
    ncvslideio.imshow(window_name, abs_dst)
    ncvslideio.waitKey(0)
    # [display]

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])
