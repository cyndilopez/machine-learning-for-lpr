############################ augment license plate images and find contours to segment characters
############################## save detected regions in directory
import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

PATH_WRITE = '/Users/cylopez/Documents/projects/license-plate-recognition/ca_pm_characters'
PATH_READ = '/Users/cylopez/Documents/projects/license-plate-recognition/plates_mania_lp/california/extracted'

def verifySize(height_img, contour):
    # char sizes are 45x77
    # char sizes are 2.5"x2.5*(3 to 4)
    x, y, w, h = cv2.boundingRect(contour)
    char_to_img_aspect = 40/224
    aspect = 2.5/(2.5*3)
    charAspect = w/h
    error = 0.6
    minHeight = 30
    minAspect = 0.2
    maxAspect = aspect+aspect*error

    if charAspect > minAspect and charAspect < maxAspect and h/height_img > char_to_img_aspect:
        return True
    else:
        return False

numImg = 0
traces = os.listdir(PATH_READ)
for jj, trace in enumerate(traces):
    if not trace.startswith('.'):
        print(trace)
        input = cv2.imread(PATH_READ +'/' + trace)
        img_copy = cv2.imread(PATH_READ +'/' + trace)
        img_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        # img_blur = cv2.blur(img_gray, (3,3),0)
        img_equ = cv2.equalizeHist(img_gray)
        ddepth = cv2.CV_8U
        height_img = input.shape[0]        
        ret, th = cv2.threshold(img_equ,80,255,cv2.THRESH_BINARY_INV)
        img_contours = copy.copy(th)
        contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        contours_after_size_verification = []
        for contour in contours:
            if verifySize(height_img, contour):
                contours_after_size_verification.append(contour)

        for contour in contours_after_size_verification:

            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            center, size, angle = rect[0], rect[1], rect[2]
            center, size = tuple(map(int, center)), tuple(map(int, size))
            width_contour = int(rect[1][1])
            height_contour = int(rect[1][0])
            height, width = img_contours.shape[0], img_contours.shape[1]
            M = cv2.getRotationMatrix2D(center, angle, 1)
            paddingw = int(width_contour/3)
            paddingh = int(height_contour/4)
            img_rot = cv2.warpAffine(img_contours, M, (width, height))
            img_crop = cv2.getRectSubPix(img_rot, (height_contour+paddingh, width_contour+paddingw), center)            
            resized_image = cv2.resize(img_crop,(34,85))

            if "image" not in trace:
                file_name = trace[:-6]+'_'+str(numImg)+'.png'
            elif "image" in trace:
                file_name = 'lp_' + trace[:-4] + '_' + str(numImg) + '.png'
            else:
                print("image not being segmented: ", trace)
                break
            
            cv2.imwrite("{}/{}".format(PATH_WRITE, file_name),resized_image)

            numImg += 1 
            print("number character saved: ", numImg)
