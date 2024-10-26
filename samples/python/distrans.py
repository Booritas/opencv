#!/usr/bin/env python

'''
Distance transform sample.

Usage:
  distrans.py [<image>]

Keys:
  ESC   - exit
  v     - toggle voronoi mode
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from common import make_cmap

def main():
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'fruits.jpg'

    fn = ncvslideio.samples.findFile(fn)
    img = ncvslideio.imread(fn, ncvslideio.IMREAD_GRAYSCALE)
    if img is None:
        print('Failed to load fn:', fn)
        sys.exit(1)

    cm = make_cmap('jet')
    need_update = True
    voronoi = False

    def update(dummy=None):
        global need_update
        need_update = False
        thrs = ncvslideio.getTrackbarPos('threshold', 'distrans')
        mark = ncvslideio.Canny(img, thrs, 3*thrs)
        dist, labels = ncvslideio.distanceTransformWithLabels(~mark, ncvslideio.DIST_L2, 5)
        if voronoi:
            vis = cm[np.uint8(labels)]
        else:
            vis = cm[np.uint8(dist*2)]
        vis[mark != 0] = 255
        ncvslideio.imshow('distrans', vis)

    def invalidate(dummy=None):
        global need_update
        need_update = True

    ncvslideio.namedWindow('distrans')
    ncvslideio.createTrackbar('threshold', 'distrans', 60, 255, invalidate)
    update()


    while True:
        ch = ncvslideio.waitKey(50)
        if ch == 27:
            break
        if ch == ord('v'):
            voronoi = not voronoi
            print('showing', ['distance', 'voronoi'][voronoi])
            update()
        if need_update:
            update()

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    ncvslideio.destroyAllWindows()
