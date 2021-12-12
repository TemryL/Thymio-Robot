# Taken from https://gist.github.com/bajcmartinez/67a47d616e1805b81e54f4724358b8fe
import cv2 as cv
import sys
from thymio_state import get_thymio_state

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

if __name__ == '__main__' :
    # Set up tracker.
    tracker = tracker_init(type=7)
    
    # Read video
    video = cv.VideoCapture(2) # for using CAM
    
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()
    
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    # Select bounding box
    bbox = cv.selectROI(frame, False)
    cv.destroyAllWindows()
    
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
        
        # Update tracker
        ok, bbox = tracker.update(frame)
        if ok:
            # Tracking success
            pass
        else :
            # Tracking failure
            cv.putText(frame, "Tracking failure detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        
        # Get state of Thymio :
        state_check, angle, center = get_thymio_state(frame, bbox)
        if state_check:
            # Display result
            cv.imshow("Tracking", frame)
        
        # Exit if ESC pressed
        if cv.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
            break
    
    video.release()
    cv.destroyAllWindows()