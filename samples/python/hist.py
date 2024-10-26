#!/usr/bin/env python

''' This is a sample for histogram plotting for RGB images and grayscale images for better understanding of colour distribution

Benefit : Learn how to draw histogram of images
          Get familier with ncvslideio.calcHist, ncvslideio.equalizeHist,ncvslideio.normalize and some drawing functions

Level : Beginner or Intermediate

Functions : 1) hist_curve : returns histogram of an image drawn as curves
            2) hist_lines : return histogram of an image drawn as bins ( only for grayscale images )

Usage : python hist.py <image_file>

Abid Rahman 3/14/12 debug Gary Bradski
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

bins = np.arange(256).reshape(256,1)

def hist_curve(im):
    h = np.zeros((300,256,3))
    if len(im.shape) == 2:
        color = [(255,255,255)]
    elif im.shape[2] == 3:
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = ncvslideio.calcHist([im],[ch],None,[256],[0,256])
        ncvslideio.normalize(hist_item,hist_item,0,255,ncvslideio.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist)))
        ncvslideio.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y

def hist_lines(im):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print("hist_lines applicable only for grayscale images")
        #print("so converting image to grayscale for representation"
        im = ncvslideio.cvtColor(im,ncvslideio.COLOR_BGR2GRAY)
    hist_item = ncvslideio.calcHist([im],[0],None,[256],[0,256])
    ncvslideio.normalize(hist_item,hist_item,0,255,ncvslideio.NORM_MINMAX)
    hist = np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        ncvslideio.line(h,(x,0),(x,y[0]),(255,255,255))
    y = np.flipud(h)
    return y


def main():
    import sys

    if len(sys.argv)>1:
        fname = sys.argv[1]
    else :
        fname = 'lena.jpg'
        print("usage : python hist.py <image_file>")

    im = ncvslideio.imread(ncvslideio.samples.findFile(fname))

    if im is None:
        print('Failed to load image file:', fname)
        sys.exit(1)

    gray = ncvslideio.cvtColor(im,ncvslideio.COLOR_BGR2GRAY)


    print(''' Histogram plotting \n
    Keymap :\n
    a - show histogram for color image in curve mode \n
    b - show histogram in bin mode \n
    c - show equalized histogram (always in bin mode) \n
    d - show histogram for gray image in curve mode \n
    e - show histogram for a normalized image in curve mode \n
    Esc - exit \n
    ''')

    ncvslideio.imshow('image',im)
    while True:
        k = ncvslideio.waitKey(0)
        if k == ord('a'):
            curve = hist_curve(im)
            ncvslideio.imshow('histogram',curve)
            ncvslideio.imshow('image',im)
            print('a')
        elif k == ord('b'):
            print('b')
            lines = hist_lines(im)
            ncvslideio.imshow('histogram',lines)
            ncvslideio.imshow('image',gray)
        elif k == ord('c'):
            print('c')
            equ = ncvslideio.equalizeHist(gray)
            lines = hist_lines(equ)
            ncvslideio.imshow('histogram',lines)
            ncvslideio.imshow('image',equ)
        elif k == ord('d'):
            print('d')
            curve = hist_curve(gray)
            ncvslideio.imshow('histogram',curve)
            ncvslideio.imshow('image',gray)
        elif k == ord('e'):
            print('e')
            norm = ncvslideio.normalize(gray, gray, alpha = 0,beta = 255,norm_type = ncvslideio.NORM_MINMAX)
            lines = hist_lines(norm)
            ncvslideio.imshow('histogram',lines)
            ncvslideio.imshow('image',norm)
        elif k == 27:
            print('ESC')
            ncvslideio.destroyAllWindows()
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    ncvslideio.destroyAllWindows()
