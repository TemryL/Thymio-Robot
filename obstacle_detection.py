import cv2 as cv
import numpy as np
import imutils
import math

THYMIO_WIDTH = 11*10**(-2)

class Polygon:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges

def color_quantization(image, k):   # From OpenCV documentation
    Z = image.reshape((-1,3))
    
    # convert to np.float32
    Z = np.float32(Z)
    
    # define criteria and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv.kmeans(Z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((image.shape))

def expand_contour(contour, m2pix_coeff):
    size = THYMIO_WIDTH*m2pix_coeff
    
    # Map coordinates of contours where 0,0 is in center of contour :
    M = cv.moments(contour)
    if M["m00"] == 0 :
        centroid = [0, 0]
        print("Warning : red contour is probably not well defined")
        color = (255, 0, 0)
    else :   
        centroid = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
        color = (0, 255, 0)
    contour = contour - centroid 
    
    new_contour = np.zeros(np.shape(contour))
    for i in range(np.shape(new_contour)[0]):
        angle = math.atan2(contour[i,0,1] , contour[i,0,0])
        new_contour[i,0,0] = contour[i,0,0] + size*math.cos(angle) 
        new_contour[i,0,1] = contour[i,0,1] + size*math.sin(angle)
    
    # Map contour back to previous coordinates by adding back position of the center
    new_contour = new_contour + centroid
    
    # Convert to int 
    new_contour = np.ceil(new_contour).astype(np.int)
    return new_contour, color

def parse_contours(contours):
    obstacles = [] 
    i = 0
    for c in contours :
        c_edges = []
        c = np.squeeze(c)
        for j in range(np.shape(c)[0]-1):
            c_edges.append((i+j, i+j+1))
        c_edges.append((i+j+1, i))
        poly = Polygon(list(map(tuple, c)), c_edges)
        obstacles.append(poly)
        i = i + j + 2
    return obstacles

def find_obstacles(img, number_of_colors, m2pix_coeff):
    image = img.copy()
    
    # Blur the image with average kernel 
    image = cv.blur(image, (11, 11))
    
    # Pixel clustering using K-means
    image = color_quantization(image, number_of_colors)
    
    # Convert image to grayscale 
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Blur slightly the image 
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    
    # Edges detection
    edges = cv.Canny(blurred, 100, 200)
    
    # Find contours from edges  
    cnts = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Approximate and expand the contour
    approx = []
    for c in cnts:
        peri = cv.arcLength(c, True)
        c = cv.approxPolyDP(c, 0.01 * peri, True)   # Approximate contour in polygon to reduce number of nodes 
        approx.append(cv.convexHull(c))     # Create convex hull of the polygon to avoid concave object in visibility graph
        approx[-1], color = expand_contour(approx[-1], m2pix_coeff)
        cv.drawContours(img, [approx[-1]], -1, color, 2) # not necessarry, only to check contour expansion
    
    # Parse contours of obstacle 
    obstacles = parse_contours(approx)
    return obstacles


