#!/usr/bin/env python

'''
Wiener deconvolution.

Sample shows how DFT can be used to perform Weiner deconvolution [1]
of an image with user-defined point spread function (PSF)

Usage:
  deconvolution.py  [--circle]
      [--angle <degrees>]
      [--d <diameter>]
      [--snr <signal/noise ratio in db>]
      [<input image>]

  Use sliders to adjust PSF paramitiers.
  Keys:
    SPACE - switch btw linear/circular PSF
    ESC   - exit

Examples:
  deconvolution.py --angle 135 --d 22  licenseplate_motion.jpg
    (image source: http://www.topazlabs.com/infocus/_images/licenseplate_compare.jpg)

  deconvolution.py --angle 86 --d 31  text_motion.jpg
  deconvolution.py --circle --d 19  text_defocus.jpg
    (image source: compact digital photo camera, no artificial distortion)


[1] http://en.wikipedia.org/wiki/Wiener_deconvolution
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# local module
from common import nothing


def blur_edge(img, d=31):
    h, w  = img.shape[:2]
    img_pad = ncvslideio.copyMakeBorder(img, d, d, d, d, ncvslideio.BORDER_WRAP)
    img_blur = ncvslideio.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)

def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = ncvslideio.warpAffine(kern, A, (sz, sz), flags=ncvslideio.INTER_CUBIC)
    return kern

def defocus_kernel(d, sz=65):
    kern = np.zeros((sz, sz), np.uint8)
    ncvslideio.circle(kern, (sz, sz), d, 255, -1, ncvslideio.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern


def main():
    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['circle', 'angle=', 'd=', 'snr='])
    opts = dict(opts)
    try:
        fn = args[0]
    except:
        fn = 'licenseplate_motion.jpg'

    win = 'deconvolution'

    img = ncvslideio.imread(ncvslideio.samples.findFile(fn), ncvslideio.IMREAD_GRAYSCALE)
    if img is None:
        print('Failed to load file:', fn)
        sys.exit(1)

    img = np.float32(img)/255.0
    ncvslideio.imshow('input', img)

    img = blur_edge(img)
    IMG = ncvslideio.dft(img, flags=ncvslideio.DFT_COMPLEX_OUTPUT)

    defocus = '--circle' in opts

    def update(_):
        ang = np.deg2rad( ncvslideio.getTrackbarPos('angle', win) )
        d = ncvslideio.getTrackbarPos('d', win)
        noise = 10**(-0.1*ncvslideio.getTrackbarPos('SNR (db)', win))

        if defocus:
            psf = defocus_kernel(d)
        else:
            psf = motion_kernel(ang, d)
        ncvslideio.imshow('psf', psf)

        psf /= psf.sum()
        psf_pad = np.zeros_like(img)
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf
        PSF = ncvslideio.dft(psf_pad, flags=ncvslideio.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
        PSF2 = (PSF**2).sum(-1)
        iPSF = PSF / (PSF2 + noise)[...,np.newaxis]
        RES = ncvslideio.mulSpectrums(IMG, iPSF, 0)
        res = ncvslideio.idft(RES, flags=ncvslideio.DFT_SCALE | ncvslideio.DFT_REAL_OUTPUT )
        res = np.roll(res, -kh//2, 0)
        res = np.roll(res, -kw//2, 1)
        ncvslideio.imshow(win, res)

    ncvslideio.namedWindow(win)
    ncvslideio.namedWindow('psf', 0)
    ncvslideio.createTrackbar('angle', win, int(opts.get('--angle', 135)), 180, update)
    ncvslideio.createTrackbar('d', win, int(opts.get('--d', 22)), 50, update)
    ncvslideio.createTrackbar('SNR (db)', win, int(opts.get('--snr', 25)), 50, update)
    update(None)

    while True:
        ch = ncvslideio.waitKey()
        if ch == 27:
            break
        if ch == ord(' '):
            defocus = not defocus
            update(None)

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    ncvslideio.destroyAllWindows()
