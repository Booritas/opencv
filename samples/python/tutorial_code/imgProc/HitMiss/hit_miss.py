import cv2 as cv
import numpy as np

input_image = np.array((
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 255, 255, 255, 0, 0, 0, 255],
    [0, 255, 255, 255, 0, 0, 0, 0],
    [0, 255, 255, 255, 0, 255, 0, 0],
    [0, 0, 255, 0, 0, 0, 0, 0],
    [0, 0, 255, 0, 0, 255, 255, 0],
    [0,255, 0, 255, 0, 0, 255, 0],
    [0, 255, 255, 255, 0, 0, 0, 0]), dtype="uint8")

kernel = np.array((
        [0, 1, 0],
        [1, -1, 1],
        [0, 1, 0]), dtype="int")

output_image = ncvslideio.morphologyEx(input_image, ncvslideio.MORPH_HITMISS, kernel)

rate = 50
kernel = (kernel + 1) * 127
kernel = np.uint8(kernel)

kernel = ncvslideio.resize(kernel, None, fx = rate, fy = rate, interpolation = ncvslideio.INTER_NEAREST)
cv.imshow("kernel", kernel)
cv.moveWindow("kernel", 0, 0)

input_image = ncvslideio.resize(input_image, None, fx = rate, fy = rate, interpolation = ncvslideio.INTER_NEAREST)
cv.imshow("Original", input_image)
cv.moveWindow("Original", 0, 200)

output_image = ncvslideio.resize(output_image, None , fx = rate, fy = rate, interpolation = ncvslideio.INTER_NEAREST)
cv.imshow("Hit or Miss", output_image)
cv.moveWindow("Hit or Miss", 500, 200)

cv.waitKey(0)
cv.destroyAllWindows()
