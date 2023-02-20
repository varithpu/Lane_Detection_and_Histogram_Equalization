#ENPM673 Project2 Problem1
from pickletools import uint8
import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

# function for creating histrogram from an image
def histogram(img):
    intensity = list(range(256))
    count = [0]*256
    # list_hist = []
    for i in range(img.shape[1]): # x-axis
        for j in range(img.shape[0]): # y-axis
            count[img[j,i]] += 1
            # list_hist.append(img[j,i])
    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    # ax.bar(intensity,count)
    # ax.hist(list_hist, bins = 256)
    # plt.show()
    return count

# function for creating histrogram from an image with tiles
def histogramAdaptive(img,tiles_x,tiles_y):
    n = int(img.shape[1]/tiles_x) # x-axis
    m = int(img.shape[0]/tiles_y) # y-axis
    count = np.zeros((tiles_y,tiles_x,256))
    for u in range(tiles_x): # tile x-axis
        for v in range(tiles_y): # tile y-axis
            for i in range(n): # image x-axis
                for j in range(m): # image y-axis
                    count[v,u,img[v*m+j,u*n+i]] += 1
    return count

# function for creating CDF from a histrogram
def CDF(count,N):
    cdf = []
    temp = 0
    for i in range(len(count)):
        temp = temp + count[i]
        cdf.append(temp/N)    
    # intensity = list(range(256))
    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    # ax.bar(intensity,cdf)
    # plt.show()
    return cdf     

# function for creating image with histogram equalization from CDF
def histEqualize(cdf,img):
    img_equalized = np.zeros((img.shape[0],img.shape[1]))
    for i in range(img.shape[1]): # x-axis
        for j in range(img.shape[0]): # y-axis
            img_equalized[j,i] = cdf[img[j,i]]*255
    return np.uint8(img_equalized)

# function for creating image with adaptive histogram equalization from CDF
def histEqualizeAdaptive(cdf_adaptive,img):
    img_equalized = np.zeros((img.shape[0],img.shape[1]))
    tiles_x = cdf_adaptive.shape[1]
    tiles_y = cdf_adaptive.shape[0]
    n = int(img.shape[1]/tiles_x) # x-axis
    m = int(img.shape[0]/tiles_y) # y-axis
    for u in range(tiles_x): # tile x-axis
        for v in range(tiles_y): # tile y-axis
            for i in range(n): # image x-axis
                for j in range(m): # image y-axis
                    img_equalized[v*m+j,u*n+i] = cdf_adaptive[v,u,img[v*m+j,u*n+i]]*255
    return np.uint8(img_equalized)

# function for limiting contrast of an image 
def contrastLimit(count, limit):
    total = 0    
    count_limit = []
    for bin in count:
        if bin > limit:
            total += bin - limit
            count_limit.append(limit)
        else:
            count_limit.append(bin)
    for bin in count_limit:
        bin += total/len(count_limit)
    return count_limit
            
# import image from file
images = []
for file in os.listdir('ENPM673/Project2/problem1_dataset'): # cheange to your file directory
    img = cv.imread(os.path.join('ENPM673/Project2/problem1_dataset',file)) # cheange to your file directory
    if img is not None:
            images.append(img)          
# main loop
for img in images:
    # convert image to gray scale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ## histogram equalization
    # create histrogram from image
    count = histogram(img_gray)
    N = img.shape[0]*img.shape[1]
    # create CDF function from histogram
    cdf = CDF(count,N)
    # equalize image with CDF function
    img_equalized = histEqualize(cdf,img_gray)
    ## histogram equalization + contrast limiting
    # limit bin size to 6000
    count_limit = contrastLimit(count, 6000)
    cdf_limit = CDF(count_limit,N)
    img_equalized_limit = histEqualize(cdf_limit,img_gray)
    ## adaptive histogram equalization
    # set number of tiles in x and y axis
    tiles_x = int(img.shape[1]/48)
    tiles_y = int(img.shape[0]/40)
    # create adaptive histrogram from image
    count_adaptive = histogramAdaptive(img_gray,tiles_x,tiles_y)
    N_adaptive = (img.shape[0]/tiles_y)*(img.shape[1]/tiles_x)
    # create CDF function for each tiles
    cdf_adaptive = np.zeros((tiles_y,tiles_x,256))
    for i in range(tiles_x):
        for j in range(tiles_y):
            cdf_adaptive[j,i] = CDF(count_adaptive[j,i],N_adaptive)
    # equalize each tiles on image with their CDF functions
    img_equalized_adaptive = histEqualizeAdaptive(cdf_adaptive,img_gray)
    ## adaptive histogram equalization + contrast limiting
    cdf_adaptive_limit = np.zeros((tiles_y,tiles_x,256))
    count_adaptive_limit = np.zeros((tiles_y,tiles_x,256))
    for i in range(tiles_x):
        for j in range(tiles_y):
            # limit bin size of each tile to 1000
            count_adaptive_limit[j,i] = contrastLimit(count_adaptive[j,i],1000)            
            cdf_adaptive_limit[j,i] = CDF(count_adaptive_limit[j,i],N_adaptive)
    img_equalized_adaptive_limit = histEqualizeAdaptive(cdf_adaptive_limit,img_gray)
    
    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    # intensity = list(range(256))   
    # count_0 = histogram(img_equalized_limit)
    # ax.bar(intensity,count_0)
    # plt.show()
    
    ## display result
    # try different outputs : img_equalized, img_equalized_limit, img_equalized_adaptive, img_equalized_adaptive_limit
    cv.imshow('Video', img_equalized_adaptive_limit) 
    cv.waitKey(20)