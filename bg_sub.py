from __future__ import print_function
import cv2 as cv
from yyz_bg_sub import createBackgroundSubtractorYYZ

import argparse
parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
parser.add_argument('--blur', type=int, default=0)
args = parser.parse_args()


if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
elif args.algo == 'KNN':
    backSub = cv.createBackgroundSubtractorKNN()
elif args.algo == 'YYZ':
    backSub = createBackgroundSubtractorYYZ()
else:
    raise NotImplemented

capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened:
    print('Unable to open: ' + args.input)
    exit(0)
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    if args.blur >= 3:
        frame_blured = cv.medianBlur(frame, args.blur)
        fgMask = backSub.apply(frame_blured)
    else:
        fgMask = backSub.apply(frame)
    
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
