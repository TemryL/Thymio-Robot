import cv2 as cv
import numpy as np
from math import atan2, pi

def find_thymio_contour(frame, thymio_box):
    bbox = thymio_box
    # Convert image to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # blur slightly the image 
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # edges detection
    edges = cv.Canny(blurred, 100, 200)
    
    # Convert image to binary
    _, bw = cv.threshold(edges, 50, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    
    # Set all pixel outside of thymio box to 0 :
    mask = np.zeros(frame.shape[:2],np.uint8)
    mask[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = 1
    bw = cv.bitwise_and(bw, bw ,mask = mask)
    
    # Find the contours of the thymio :
    contours, hierarchy = cv.findContours(bw, cv.RETR_TREE , cv.CHAIN_APPROX_SIMPLE)
    thymio_contour = []
    shape_contour = []
    for i in range(len(contours)):
        if (hierarchy[0,i,2] != -1 and hierarchy[0,i,3] == -1) :
            thymio_contour.append(contours[i])
        if (hierarchy[0,i,2] == -1 and hierarchy[0,i,3] != -1) :
            shape_contour.append(contours[i])
    
    # # Find the contours of the thymio :
    # contours, hierarchy = cv.findContours(bw, cv.RETR_EXTERNAL , cv.CHAIN_APPROX_SIMPLE)
    # thymio_contour = []
    # thymio_contour.append(contours[0])
    
    # contours, hierarchy = cv.findContours(bw, cv.RETR_TREE , cv.CHAIN_APPROX_SIMPLE)
    # shape_contour = []
    # for i in range(len(contours)):
    #     if (hierarchy[0,i,2] == -1 and hierarchy[0,i,3] != -1) :
    #         shape_contour.append(contours[i])
    
    # Check if contours are well defined :
    if (len(thymio_contour) != 1) or (len(shape_contour) != 1):
        ok = False
    else :
        thymio_contour = thymio_contour[0]
        shape_contour = shape_contour[0]
        ok = True    
    return ok, thymio_contour, shape_contour
    
    # contours, hierarchy = cv.findContours(bw, cv.RETR_TREE , cv.CHAIN_APPROX_SIMPLE)
    # triangle_contour = []
    # for c in contours :
    #     peri = cv.arcLength(c, True)
    #     approx = cv.approxPolyDP(c, 0.07 * peri, True)
    #     if len(approx) == 3 :
    #         triangle_contour.append(approx)
    # # Check if contours are well defined :
    # if (len(triangle_contour) != 1) :
    #     ok = False
    # else :
    #     triangle_contour = triangle_contour[0]
    #     ok = True    
    
    # return ok, triangle_contour

class Point:
    def __init__(self, coords):
        self.x = coords[0]
        self.y = coords[1]

def get_angle_and_center(thymio_contour, shape_contour):
    # Compute centroid of the Thymio :
    M = cv.moments(thymio_contour)
    center =  [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
    
    # Compute centroid of the black rectangle :
    M = cv.moments(shape_contour)
    shape_pos =  [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
    
    # Compute angle :
    v = [(center[0] - shape_pos[0]), (center[1] - shape_pos[1])]
    angle = atan2(v[1], v[0])
    
    return angle, center, shape_pos

def get_thymio_state(frame, thymio_box):
    ok, thymio_contour, shape_contour = find_thymio_contour(frame, thymio_box)
    if ok :
        angle, center, shape_pos = get_angle_and_center(thymio_contour, shape_contour)
        # Draw contour of thymio and centers only for visualisation purposes
        cv.drawContours(frame, [thymio_contour], 0, (0, 0, 255), 2)
        cv.circle(frame, center, 3, (0, 0, 255), 2)
        cv.circle(frame, shape_pos, 3, (255, 0, 255), 2)
    else :
        angle = []
        center = []
    return ok, angle, center

# def get_angle_and_center(triangle_contour):
#     # Compute centroid of the Thymio :
#     center =  np.mean(np.squeeze(triangle_contour), 0).astype(int)
    
#     triangle = []
#     for i in range(len(triangle_contour)):
#         triangle.append(np.squeeze(triangle_contour)[i,:])
    
#     seg = []
#     for i in range(2):
#         seg.append(np.linalg.norm(triangle[i+1]-triangle[i]))
#     seg.append(np.linalg.norm(triangle[0]-triangle[i+1]))
#     idx = np.argmax(seg)
    
#     if idx == 0 :
#         top = np.squeeze(triangle_contour[2])
#     if idx == 1 :
#         top = np.squeeze(triangle_contour[0])
#     if idx == 2 :
#         top = np.squeeze(triangle_contour[1])
    
#     # Compute angle :
#     v = [(center[0] - top[0]), (center[1] - top[1])]
#     angle = atan2(v[1], v[0])
    
#     return angle, center, top

# def get_thymio_state(frame, thymio_box):
#     ok, triangle_contour = find_thymio_contour(frame, thymio_box)
#     if ok :
#         angle, center, top = get_angle_and_center(triangle_contour)
#         # Draw contour of thymio and centers only for visualisation purposes
#         cv.drawContours(frame, [triangle_contour], 0, (0, 0, 255), 2)
#         cv.circle(frame, center, 3, (0, 0, 255), 2)
#         cv.circle(frame, top, 3, (255, 0, 255), 2)
#     else :
#         angle = []
#         center = []
#     return ok, angle, center

def update_thymio_state(frame, bbox, thymio_state):
    # Get state of Thymio :
    state_check, angle, center = get_thymio_state(frame, bbox)
    if state_check:
        # Remove outlier state :
        angle_tol = 0.25
        if abs((angle - thymio_state[-1][2])) > angle_tol:
            angle = thymio_state[-1][2]
        thymio_state.append((center[0], center[1], angle))