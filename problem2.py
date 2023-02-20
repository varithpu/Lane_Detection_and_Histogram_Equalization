#ENPM673 Project2 Problem2
import cv2 as cv
import numpy as np
import copy
import math
import copy

# function for Hough Transform
def houghTransform(edges):
    # initialize
    m = edges.shape[1] # x-axis
    n = edges.shape[0] # y-axis
    dmax = int(math.sqrt(n*n+m*m))
    H = np.zeros((2*dmax,180)) # H[d,zeta]
    # main Haugh Transform loop
    for i in range(m): # x-axis
        for j in range(n): # y-axis
            if edges[j,i] == 255:
                for deg in range(180):
                    d = int(i*math.cos(math.radians(deg)) + j*math.sin(math.radians(deg))) # calculate d in [d,zeta] domain
                    H[d+dmax,deg] += 1
    # set threshold for black out
    th_s = 5
    th_l = 500
    # get index of lane 1 first line
    H_1 = copy.deepcopy(H)
    ind_1_1 = np.argwhere(H_1 == H_1.max())
    ind_1_1_max = H_1.max()
    # get index of lane 2 first line
    H_1[ind_1_1[0,0]-th_s:ind_1_1[0,0]+th_s,ind_1_1[0,1]-th_s:ind_1_1[0,1]+th_s] = 0
    H_2 = copy.deepcopy(H)
    H_2[ind_1_1[0,0]-th_l:ind_1_1[0,0]+th_l,ind_1_1[0,1]-th_l:ind_1_1[0,1]+th_l] = 0
    ind_2_1 = np.argwhere(H_2 == H_2.max())
    ind_2_1_max = H_2.max()
    # get index of lane 1 second line
    H_1[ind_2_1[0,0]-th_l:ind_2_1[0,0]+th_l,ind_2_1[0,1]-th_l:ind_2_1[0,1]+th_l] = 0
    ind_1_2 = np.argwhere(H_1 == H_1.max())
    ind_1_2_max = H_1.max()
    # get index of lane 2 second line 
    H_2[ind_2_1[0,0]-th_s:ind_2_1[0,0]+th_s,ind_2_1[0,1]-th_s:ind_2_1[0,1]+th_s] = 0
    ind_2_2 = np.argwhere(H_2 == H_2.max())
    ind_2_2_max = H_2.max()
    # average voting
    ind_1_max = (ind_1_1_max + ind_1_2_max)/2
    ind_2_max = (ind_2_1_max + ind_2_2_max)/2
    # average line of lane1
    ind_1 = np.zeros((1,2))
    ind_1[0,0] = (ind_1_1[0,0] + ind_1_2[0,0])/2 - dmax
    ind_1[0,1] = (ind_1_1[0,1] + ind_1_2[0,1])/2
    # average line of lane2
    ind_2 = np.zeros((1,2))
    ind_2[0,0] = (ind_2_1[0,0] + ind_2_2[0,0])/2 - dmax
    ind_2[0,1] = (ind_2_1[0,1] + ind_2_2[0,1])/2
    return ind_1, ind_1_max, ind_2, ind_2_max    

# import video
capture = cv.VideoCapture('ENPM673/Project2/whiteline.mp4') # cheange to your video directory
count = 0
ind_1_col_prev = None
ind_2_col_prev = None
# main loop
while True:
        isTrue, frame = capture.read()
        if isTrue == True: 
             # convert video frame to gray scale and do thresholding
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _,frame_thresh = cv.threshold(frame_gray,127,255,cv.THRESH_BINARY)
            # mask image and keep only interested areas
            xmax = frame.shape[1]
            ymax = frame.shape[0]
            y0 = int(6*frame.shape[0]/10)
            poly_mask = np.zeros((ymax,xmax), np.uint8)
            pts = np.array([[0,ymax],[450,y0],[xmax-450,y0],[xmax,ymax]], np.int32)
            cv.fillConvexPoly(poly_mask,(pts),255)
            frame_masked = cv.bitwise_and(frame_thresh,frame_thresh,mask = poly_mask)   
            # detect edges with canny
            edges = cv.Canny(frame_masked,100,200)
            # find lanes with Hough transformation
            ind_1, ind_1_max, ind_2, ind_2_max = houghTransform(edges)
            # average maximum Hough Transform voting from 2 consecutive frame
            if count == 0:
                ind_1_max_prev = ind_1_max
                ind_2_max_prev = ind_2_max
                ind_1_max_avg = ind_1_max
                ind_2_max_avg = ind_2_max
            else:
                ind_1_max_avg = (ind_1_max_prev + ind_1_max)/2
                ind_2_max_avg = (ind_2_max_prev + ind_2_max)/2
                ind_1_max_prev = ind_1_max
                ind_2_max_prev = ind_2_max
            # determine lane color by average voting
            if ind_1_max_avg > 150:
                ind_1_col = (0,255,0)
            else:
                ind_1_col = (0,0,255)
            if ind_2_max_avg > 150:
                ind_2_col = (0,255,0)
            else:
                ind_2_col = (0,0,255)
            # plot lane 1
            edges_rgb = cv.cvtColor(edges,cv.COLOR_GRAY2RGB)
            d_1 = ind_1[0,0]
            deg_1 = ind_1[0,1]
            x0_1 = int((d_1-y0*math.sin(math.radians(deg_1)))/math.cos(math.radians(deg_1)))
            x1_1 = int((d_1-ymax*math.sin(math.radians(deg_1)))/math.cos(math.radians(deg_1)))
            cv.line(edges_rgb,(x0_1,y0),(x1_1,ymax),ind_1_col,3)
            cv.line(frame,(x0_1,y0),(x1_1,ymax),ind_1_col,3)
            # plot lane 2
            d_2 = ind_2[0,0]
            deg_2 = ind_2[0,1]
            x0_2 = int((d_2-y0*math.sin(math.radians(deg_2)))/math.cos(math.radians(deg_2)))
            x1_2 = int((d_2-ymax*math.sin(math.radians(deg_2)))/math.cos(math.radians(deg_2)))
            cv.line(edges_rgb,(x0_2,y0),(x1_2,ymax),ind_2_col,3)
            cv.line(frame,(x0_2,y0),(x1_2,ymax),ind_2_col,3)
            # show frame
            cv.imshow('Video', frame) # try different outputs: frame, edges_rgb
            cv.waitKey(1)
            count += 1
        else:
            break