############################ augment license plate images and find contours to segment characters
############################## save detected regions in directory
import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

PATH_WRITE = '/Users/cylopez/Documents/projects/license-plate-recognition/ca_pm_characters'
# PATH_READ = '/Users/cylopez/Documents/projects/license-plate-recognition/plates_mania_lp/california/extracted'
PATH_READ = '/Users/cylopez/Documents/projects/license-plate-recognition/ca_plates_usimg_scraped'

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
        input = cv2.imread(PATH_READ +'/' + trace)
        img_copy = cv2.imread(PATH_READ +'/' + trace)
        try: 
            img_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        
        except:
            print(trace)
            continue
            
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

            # bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            paddingw = int(w/3)
            paddingh = int(h/4)
            #img_crop used to reference copy of input, and the written file was img crop
            img_crop = cv2.getRectSubPix(img_contours, (w+paddingw,h+paddingh), (x+w/2,y+h/2))
            resized_image = cv2.resize(img_crop,(34,85))

            if "image" not in trace:
                # file_name = trace[:-6]+'_'+str(numImg)+'.png'
                file_name = trace[:-4] + '_' + str(numImg) + '.png'
            elif "image" in trace:
                # file_name = 'lp_' + trace[:-4] + '_' + str(numImg) + '.png'
                file_name = trace[:-4] + '_' + str(numImg) + '.png'
            else:
                print("image not being segmented: ", trace)
                break
            
            cv2.imwrite("{}/{}".format(PATH_WRITE, file_name),resized_image)

            numImg += 1 
            print("number character saved: ", numImg)
