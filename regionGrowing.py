#thank you to marco barreto
#github https://marcoagbarreto.github.io/VisionArtificial/programs/region_growing.html

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
%matplotlib inline

def region_Growing(img, seed, region_threshold  = 70, connectivity = 4):
    #Parameters for region growing
    region_size = 1
    pixel_distance = 0
    neighbor_points_list = []
    neighbor_intensity_list = []


    # threshold tests
    if (not isinstance(region_threshold, int)) :
        raise TypeError("(%s) Int expected!" % (sys._getframe().f_code.co_name))
    elif region_threshold < 0:
        raise ValueError("(%s) Positive value expected!" % (sys._getframe().f_code.co_name))

    # Connectivity tests
    if (not isinstance(connectivity, int)) :
        raise TypeError("(%s) Int expected!" % (sys._getframe().f_code.co_name))
    if connectivity == 4:
        neighbors = [(1, 0), (0, 1), (-1, 0), (0, -1)] # 4 connectivity
    elif connectivity == 8:
        neighbors = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)] # 8 connectivity
    else:
        neighbors = [(1, 0), (0, 1), (-1, 0), (0, -1)] # 4 connectivity

    #Mean of the segmented region
    region_mean = img[seed[0],seed[1]] # This changed

    #Input image parameters
    height, width = img.shape
    image_size = height * width

    #Initialize segmented output image
    segmented_img = np.zeros((height, width), np.uint8)

    #Region growing until intensity difference becomes greater than certain threshold
    while (pixel_distance < region_threshold) & (region_size < image_size):
        #Loop through neighbor pixels
        for i in range(len(neighbors)):
            #Compute the neighbor pixel position
            x_new = seed[0] + neighbors[i][0]
            y_new = seed[1] + neighbors[i][1]

            #Boundary Condition - check if the coordinates are inside the image
            check_inside = (x_new >= 0) and (y_new >= 0) and (x_new < height) and (y_new < width)
            #Add neighbor if inside and not already in segmented_img
            if check_inside:
                if segmented_img[x_new, y_new] == 0:
                    neighbor_points_list.append([x_new, y_new])
                    neighbor_intensity_list.append(img[x_new, y_new])
                    segmented_img[x_new, y_new] = 255

        #Add pixel with intensity nearest to the mean to the region
        distance = abs(neighbor_intensity_list-region_mean)
        pixel_distance = min(distance)
        index = np.argmin(distance)
        segmented_img[seed[0], seed[1]] = 255
        region_size += 1

        #New region mean
        region_mean = (region_mean*region_size + neighbor_intensity_list[index])/(region_size+1)

        #Update the seed value
        seed = neighbor_points_list[index]
        #Remove the value from the neighborhood lists
        del neighbor_points_list[index]
        del neighbor_intensity_list[index]

    return segmented_img
