import cv2 as cv
import numpy as np

img = ncvslideio.imread(ncvslideio.samples.findFile('sudoku.png'))
gray = ncvslideio.cvtColor(img,ncvslideio.COLOR_BGR2GRAY)
edges = ncvslideio.Canny(gray,50,150,apertureSize = 3)
lines = ncvslideio.HoughLinesP(edges,1,np.pi/180,100,minLineLength=100,maxLineGap=10)
for line in lines:
    x1,y1,x2,y2 = line[0]
    ncvslideio.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv.imwrite('houghlines5.jpg',img)
