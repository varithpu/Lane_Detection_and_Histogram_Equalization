#ENPM673 Project2 Problem3
import cv2 as cv
import numpy as np
import math
from scipy import optimize
import math
from statistics import mean

# create histrogram from an image
def histogram(img):
    count = [0]*256
    for i in range(img.shape[1]): # x-axis
        for j in range(img.shape[0]): # y-axis
            count[img[j,i]] += 1
    return count

# create CDF from a histrogram
def CDF(count,N):
    cdf = []
    temp = 0
    for i in range(len(count)):
        temp = temp + count[i]
        cdf.append(temp/N)  
    return cdf   

# create image with histogram equalization from CDF
def histEqualize(cdf,img):
    img_equalized = np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[1]): # x-axis
        for j in range(img.shape[0]): # y-axis
            img_equalized[j,i] = cdf[img[j,i]]*255
    return np.uint8(img_equalized)

# import video
capture = cv.VideoCapture('ENPM673/Project2/challenge.mp4') # cheange to your video directory
# main loop
while True:
        isTrue, frame = capture.read()
        if isTrue == True: 
            # calculate Homography 
            xmax = frame.shape[1]
            ymax = frame.shape[0]-50
            y0 = int(6*frame.shape[0]/10)
            warp_x = 500
            warp_y = 900
            pts1 = np.float32([[0,ymax],[580,y0],[xmax-580,y0],[xmax,ymax]])
            pts2 = np.float32([[0,warp_y],[0,0],[warp_x,0],[warp_x,warp_y]])
            H = cv.getPerspectiveTransform(pts1, pts2)
            # warp image according to Homography
            frame_warp = cv.warpPerspective(frame, H, (warp_x, warp_y))
            # convert frame to gray scale and do histogram equalization   
            frame_warp_gray = cv.cvtColor(frame_warp, cv.COLOR_BGR2GRAY)
            count = histogram(frame_warp_gray)
            N = frame_warp_gray.shape[0]*frame_warp_gray.shape[1]
            cdf = CDF(count,N)
            frame_equalized = histEqualize(cdf,frame_warp_gray)
            _,frame_warp_thresh = cv.threshold(frame_equalized,240,255,cv.THRESH_BINARY)
            frame_warp_rgb = cv.cvtColor(frame_warp_thresh,cv.COLOR_GRAY2RGB)
            ## detect lane L
            # detect bottom portion of lane
            lane_L = []
            for j in range(warp_y-1,warp_y-100,-1): # y-axis
                count = 0
                sum = 0
                for i in range(0,int(warp_x/2)): # x-axis
                     if frame_warp_thresh[j,i] == 255:
                         sum += i
                         count += 1
                if count != 0:
                    avg = int(sum/count)
                    lane_L.append((avg,j))
            if len(lane_L)>0:
                prev_avg = lane_L[-1][0]
            # proporgate lane connectivity upward
            count0 = 0
            for j in range(warp_y-101,0,-1): # y-axis
                count = 0
                sum = 0
                left_bound = prev_avg-50
                right_bound = prev_avg+50
                if prev_avg+50 > warp_x:
                    right_bound = warp_x
                if prev_avg-50 < 0:
                    left_bound = 0
                for i in range(left_bound,right_bound):
                    cv.line(frame_warp_rgb,(prev_avg-50,j),(prev_avg+50,j),(0,255,255),1)
                    if frame_warp_thresh[j,i] == 255:
                         sum += i
                         count += 1
                if count != 0 and count<30:
                    count0 = 0
                    avg = int(sum/count)
                    lane_L.append((avg,j))
                else:
                    count0 += 1
                if count0 >130:
                    break
                prev_avg = avg
            # estimate road raduis of curvature and origin by fitting a circle with least square  
            lane_L_x = np.empty(0)
            lane_L_y = np.empty(0)   
            for pts in lane_L:
                cv.circle(frame_warp_rgb,pts,5,(0,255,0))
                lane_L_x = np.append(lane_L_x,pts[0])
                lane_L_y = np.append(lane_L_y,pts[1])
            # function for calculating distance between the data points and the mean circle centered at c=(xc, yc)
            def func_L(c):
                xc, yc = c
                R = np.sqrt((lane_L_x-xc)**2 + (lane_L_y-yc)**2)
                return R - R.mean()
            x_L_m = np.mean(lane_L_x)
            y_L_m = np.mean(lane_L_y)
            # origin of circle
            c0 = x_L_m, y_L_m
            center_L, _ = optimize.leastsq(func_L, c0)
            xc_L, yc_L = center_L
            R_L = np.sqrt((lane_L_x-xc_L)**2 + (lane_L_y-yc_L)**2)
            # radius of circle
            R_L_m = R_L.mean()
            # square error
            error_L = np.sum((R_L - R_L_m)**2)/(len(lane_L)**2)
            # plot circle, use previous circle if error is too high or data amount is too low
            lines_disp = np.zeros((frame_warp.shape[0],frame_warp.shape[1],3),np.uint8)  
            if error_L<0.2 and len(lane_L)>150:
                xc_L_prev = xc_L
                yc_L_prev = yc_L
                R_L_m_prev = R_L_m
                cv.circle(frame_warp_rgb,(int(xc_L),int(yc_L)),int(R_L_m),(0,0,255),5)
                cv.circle(lines_disp,(int(xc_L),int(yc_L)),int(R_L_m),(0,0,255),5)   
            else:
                cv.circle(frame_warp_rgb,(int(xc_L_prev),int(yc_L_prev)),int(R_L_m_prev),(255,0,0),5)
                cv.circle(lines_disp,(int(xc_L_prev),int(yc_L_prev)),int(R_L_m_prev),(255,0,0),5)
                xc_L = xc_L_prev 
                yc_L = yc_L_prev
                R_L_m = R_L_m_prev
            # detect lanes R
            # detect bottom portion of lane
            lane_R = [] 
            for j in range(warp_y-1,warp_y-100,-1): # y-axis
                count = 0
                sum = 0
                for i in range(int(warp_x/2)+1,warp_x): # x-axis
                     if frame_warp_thresh[j,i] == 255:
                         sum += i
                         count += 1
                if count != 0 :
                    avg = int(sum/count)
                    lane_R.append((avg,j))
            prev_avg = lane_R[-1][0]
            # proporgate lane connectivity upward
            count0 = 0
            for j in range(warp_y-101,0,-1): # y-axis
                count = 0
                sum = 0
                left_bound = prev_avg-50
                right_bound = prev_avg+50
                if prev_avg+50 > warp_x:
                    right_bound = warp_x
                if prev_avg-50 < 0:
                    left_bound = 0
                for i in range(left_bound,right_bound):
                    cv.line(frame_warp_rgb,(prev_avg-50,j),(prev_avg+50,j),(0,255,255),1)
                    if frame_warp_thresh[j,i] == 255:
                         sum += i
                         count += 1
                if count != 0 and count<30:
                    count0 = 0
                    avg = int(sum/count)
                    lane_R.append((avg,j))
                else:
                    count0 += 1
                if count0 >130:
                    break
                prev_avg = avg       
            # estimate road raduis of curvature and origin by fitting a circle with least square              
            lane_R_x = np.empty(0)
            lane_R_y = np.empty(0)   
            for pts in lane_R:
                cv.circle(frame_warp_rgb,pts,5,(0,255,0))
                lane_R_x = np.append(lane_R_x,pts[0])
                lane_R_y = np.append(lane_R_y,pts[1])
             # function for calculating distance between the data points and the mean circle centered at c=(xc, yc)    
            def func_R(c):
                xc, yc = c
                R = np.sqrt((lane_R_x-xc)**2 + (lane_R_y-yc)**2)
                return R - R.mean()
            x_R_m = np.mean(lane_R_x)
            y_R_m = np.mean(lane_R_y)
            c0 = x_R_m, y_R_m
            center_R, _ = optimize.leastsq(func_R, c0)
            # origin of circle
            xc_R, yc_R = center_R
            R_R = np.sqrt((lane_R_x-xc_R)**2 + (lane_R_y-yc_R)**2)
            # radius of circle
            R_R_m = R_R.mean()
            # square error
            error_R = np.sum((R_R - R_R_m)**2)/(len(lane_R)**2)     
            # plot circle, use previous circle if error is too high or data amount is too low
            if error_R<0.2 and len(lane_R)>150:
                xc_R_prev = xc_R
                yc_R_prev = yc_R
                R_R_m_prev = R_R_m
                cv.circle(frame_warp_rgb,(int(xc_R),int(yc_R)),int(R_R_m),(0,0,255),5)
                cv.circle(lines_disp,(int(xc_R),int(yc_R)),int(R_R_m),(0,0,255),5)
            else:
                cv.circle(frame_warp_rgb,(int(xc_R_prev),int(yc_R_prev)),int(R_R_m_prev),(255,0,0),5)
                cv.circle(lines_disp,(int(xc_R_prev),int(yc_R_prev)),int(R_R_m_prev),(255,0,0),5)
                xc_R = xc_R_prev  
                yc_R = yc_R_prev 
                R_R_m = R_R_m_prev 
            # fill in lane  
            lines_disp_fill = np.zeros((frame_warp.shape[0],frame_warp.shape[1],3),np.uint8)            
            for j in range(int(lines_disp.shape[0]/10)): # y-axis       
                list = []     
                for i in range(lines_disp.shape[1]): # x-axis
                    if any(lines_disp[10*j,i] != (0,0,0)):
                        list.append(i)
                if len(list)> 10:
                    cv.line(lines_disp_fill,(list[0],10*j),(list[-1],10*j),(0,0,255),10)
            # find average radius
            xc_m = int((xc_R + xc_L)/2)
            yc_m = int((yc_R + yc_L)/2)
            R_m = int((R_R_m + R_L_m)/2)
            # create arrow at middle of the lane
            start = True
            for y in range(int(lines_disp.shape[0]/30),0,-1): # y-axis      
                y = 30*y   
                if R_m**2-(yc_m-y)**2 >= 0:
                    x = int(-math.sqrt(R_m**2-(yc_m-y)**2) + xc_m)
                    if start == True:
                        x_prev = x
                        y_prev = y
                        start = False
                    else:
                        cv.arrowedLine(lines_disp,(x_prev,y_prev),(x,y),(0,0,255),5,tipLength=0.5)
                        start = True
            # calculate width of lane in pixels
            if R_m**2-(yc_m-frame.shape[0])**2 > 0:        
                x0 = int(-math.sqrt(R_m**2-(yc_m-frame.shape[0])**2) + xc_m)    
            if R_L_m**2-(yc_L-frame.shape[0])**2 >0:    
                x0_L = int(-math.sqrt(R_L_m**2-(yc_L-frame.shape[0])**2) + xc_L)
            if R_R_m**2-(yc_R-frame.shape[0])**2 > 0:
                x0_R = int(-math.sqrt(R_R_m**2-(yc_R-frame.shape[0])**2) + xc_R)   
            # estimate car offset from lane center
            diff = x0-frame_warp.shape[1]/2
            if diff>0:
                offset = "left"
            else:
                offset = "right"
            # create turning recommendation
            if xc_m >= frame_warp.shape[1]/2 and R_m < 10000:
                turn = "Turn Right"
            elif xc_m < frame_warp.shape[1]/2 and R_m < 10000:
                turn = "Turn Right"
            else:
                turn = "Go Stright"
            # calculate scale for converting between pixels and meters
            scale = 3.7/(x0_R-x0_L)       
            # display lane status ion video
            font = cv.FONT_HERSHEY_SIMPLEX 
            cv.putText(frame, "Radius of Curvature: " + str(round(R_m*scale)) + "m", (10,25), font, 0.8, (0, 255, 0), 2)
            cv.putText(frame, str(turn), (10,60), font, 0.8, (0, 255, 0), 2)
            cv.putText(frame, "Vehicle is " + str(round(abs(diff*scale),2)) + "m to the " + offset + " of lane center", (10,95), font, 0.8, (0, 255, 0), 2)
            # calculate inverse Homography
            H_inv = np.linalg.inv(H)
            # warp lane detection to original frame
            lines_disp_fill_inv = cv.warpPerspective(lines_disp_fill, H_inv, (frame.shape[1], frame.shape[0]))
            lines_disp_inv = cv.warpPerspective(lines_disp, H_inv, (frame.shape[1], frame.shape[0]))
            # merge lane detection to original image
            blended_0 = cv.addWeighted(frame, 1, lines_disp_fill_inv, 0.1, 0.5) 
            blended_1 = cv.addWeighted(blended_0, 1, lines_disp_inv, 0.5, 0.5) 
            # display video
            cv.imshow('Video', blended_1) 
            cv.waitKey(1)
        else:
            break
        