import numpy as np
import cv2 as cv
cap = ncvslideio.VideoCapture(ncvslideio.samples.findFile("vtest.avi"))
ret, frame1 = cap.read()
prvs = ncvslideio.cvtColor(frame1, ncvslideio.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255
while(1):
    ret, frame2 = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    next = ncvslideio.cvtColor(frame2, ncvslideio.COLOR_BGR2GRAY)
    flow = ncvslideio.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = ncvslideio.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = ncvslideio.normalize(mag, None, 0, 255, ncvslideio.NORM_MINMAX)
    bgr = ncvslideio.cvtColor(hsv, ncvslideio.COLOR_HSV2BGR)
    ncvslideio.imshow('frame2', bgr)
    k = ncvslideio.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        ncvslideio.imwrite('opticalfb.png', frame2)
        ncvslideio.imwrite('opticalhsv.png', bgr)
    prvs = next

cv.destroyAllWindows()
