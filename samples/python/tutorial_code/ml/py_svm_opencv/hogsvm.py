#!/usr/bin/env python

import cv2 as cv
import numpy as np

SZ=20
bin_n = 16 # Number of bins


affine_flags = ncvslideio.WARP_INVERSE_MAP|ncvslideio.INTER_LINEAR

## [deskew]
def deskew(img):
    m = ncvslideio.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = ncvslideio.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img
## [deskew]

## [hog]
def hog(img):
    gx = ncvslideio.Sobel(img, ncvslideio.CV_32F, 1, 0)
    gy = ncvslideio.Sobel(img, ncvslideio.CV_32F, 0, 1)
    mag, ang = ncvslideio.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist
## [hog]

img = ncvslideio.imread(ncvslideio.samples.findFile('digits.png'),0)
if img is None:
    raise Exception("we need the digits.png image from samples/data here !")


cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]

# First half is trainData, remaining is testData
train_cells = [ i[:50] for i in cells ]
test_cells = [ i[50:] for i in cells]

######     Now training      ########################

deskewed = [list(map(deskew,row)) for row in train_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
trainData = np.float32(hogdata).reshape(-1,64)
responses = np.repeat(np.arange(10),250)[:,np.newaxis]

svm = ncvslideio.ml.SVM_create()
svm.setKernel(ncvslideio.ml.SVM_LINEAR)
svm.setType(ncvslideio.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)

svm.train(trainData, ncvslideio.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')

######     Now testing      ########################

deskewed = [list(map(deskew,row)) for row in test_cells]
hogdata = [list(map(hog,row)) for row in deskewed]
testData = np.float32(hogdata).reshape(-1,bin_n*4)
result = svm.predict(testData)[1]

#######   Check Accuracy   ########################
mask = result==responses
correct = np.count_nonzero(mask)
print(correct*100.0/result.size)
