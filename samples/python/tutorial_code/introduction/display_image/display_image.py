## [imports]
import cv2 as cv
import sys
## [imports]
## [imread]
img = ncvslideio.imread(ncvslideio.samples.findFile("starry_night.jpg"))
## [imread]
## [empty]
if img is None:
    sys.exit("Could not read the image.")
## [empty]
## [imshow]
cv.imshow("Display window", img)
k = ncvslideio.waitKey(0)
## [imshow]
## [imsave]
if k == ord("s"):
    ncvslideio.imwrite("starry_night.png", img)
## [imsave]
