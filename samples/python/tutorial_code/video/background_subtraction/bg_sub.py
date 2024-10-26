from __future__ import print_function
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

## [create]
#create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = ncvslideio.createBackgroundSubtractorMOG2()
else:
    backSub = ncvslideio.createBackgroundSubtractorKNN()
## [create]

## [capture]
capture = ncvslideio.VideoCapture(ncvslideio.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
## [capture]

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    ## [apply]
    #update the background model
    fgMask = backSub.apply(frame)
    ## [apply]

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    ncvslideio.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    ncvslideio.putText(frame, str(capture.get(ncvslideio.CAP_PROP_POS_FRAMES)), (15, 15),
               ncvslideio.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    ## [display_frame_number]

    ## [show]
    #show the current frame and the fg masks
    ncvslideio.imshow('Frame', frame)
    ncvslideio.imshow('FG Mask', fgMask)
    ## [show]

    keyboard = ncvslideio.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
