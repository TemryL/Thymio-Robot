import cv2 as cv
import sys
import numpy as np
import math
from threading import Timer

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()
    
    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)
    
    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True
    
    def stop(self):
        self._timer.cancel()
        self.is_running = False

def tracker_init(type):
    (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')
    
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[type]
    
    if int(minor_ver) < 3:
        tracker = cv.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
            tracker = cv.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            tracker = cv.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            tracker = cv.TrackerCSRT_create()
    return tracker

def select_goal(event, x, y, flags, params):
    global goal
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        goal = (x, y)
        cv.destroyAllWindows()

def initial_thymio_contour(frame, thymio_box):
    bbox = thymio_box
    # Convert image to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Blur slightly the image 
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Edges detection
    edges = cv.Canny(blurred, 100, 200)
    
    # Convert image to binary
    _, bw = cv.threshold(edges, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    # Set all pixel outside of thymio box to 0 :
    mask = np.zeros(frame.shape[:2],np.uint8)
    mask[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1
    bw = cv.bitwise_and(bw, bw ,mask = mask)
    
    # Find the contours of the thymio :
    contours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    idx = np.argmax([cv.contourArea(c) for c in contours])
    c = contours[idx]
    contour_length = cv.arcLength(c, True)
    
    return c

def get_start_position(thymio_contour):
    # Compute centroid of the thymio from the contour of the Thymio :
    M = cv.moments(thymio_contour)
    start =  (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return start

def initialization():
    # Set up tracker.
    tracker = tracker_init(type=7)
    
    # Read video
    video = cv.VideoCapture(2) # for using CAM
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
        
    # Read first frame.
    ok, initial_frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    # Select bounding box containing the Thymio :
    print("Select bounding box containing the Thymio")
    bbox = cv.selectROI(initial_frame, False)
    cv.destroyAllWindows()
    
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(initial_frame, bbox)
    
    # Find contour of Thymio at initial position :
    thymio_contour = initial_thymio_contour(initial_frame, bbox)
    
    # Get start position :
    start = get_start_position(thymio_contour)
    
    # Select goal :
    print("Select goal position")
    cv.imshow('Goal selection', initial_frame)
    # Setting mouse handler for the image and calling the click_event() function
    cv.setMouseCallback('Goal selection', select_goal)
    # Wait for a key to be pressed to exit
    cv.waitKey(0)  
    
    # Exctract color of background :
    print("Select bounding box containing background only")
    bg_box = cv.selectROI(initial_frame, False)
    cv.destroyAllWindows()
    background_color = cv.mean(initial_frame[bg_box[1]:bg_box[1]+bg_box[3], bg_box[0]:bg_box[0]+bg_box[2]])
    background_color = background_color[0:3]
    
    return video, tracker, ok, start, goal, initial_frame, bbox, background_color 