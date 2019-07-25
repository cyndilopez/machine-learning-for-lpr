############################ augment license plate images and find contours to segment characters
############################## save detected regions in directory
import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

PATH_WRITE = '/Users/cylopez/Documents/projects/license-plate-recognition/ca_pm_characters'
PATH_READ = '/Users/cylopez/Documents/projects/license-plate-recognition/plates_mania_lp/test'

def verifySize(height_img, contour):
    # return True
    # char sizes are 45x77
    # char sizes are 2.5"x2.5*(3 to 4)
    x, y, w, h = cv2.boundingRect(contour)
    char_to_img_aspect = 40/224
    aspect = 2.5/(2.5*3) #based on eyeballing
    charAspect = w/h
    error = 0.6
    # for image shape of (164, 287)
    minHeight = 30
    minAspect = 0.2
    maxAspect = aspect+aspect*error

    if charAspect > minAspect and charAspect < maxAspect and h/height_img > char_to_img_aspect:
        # permanently altering image by drawing rectangle
        cv2.rectangle(th,(x,y),(x+w,y+h),(255,255,255),2)
        plt.imshow(th), plt.title("Bounding Box Contours")
        plt.show()
        return True
    else:
        return False

ignored = ['characters', 'test_write']
numImg = 0
# path = './scraped_plates_in' #### modify this
# os.makedirs(path+'/'+ignored[0])
traces = os.listdir(PATH_READ)
for jj, trace in enumerate(traces):
    if not trace.startswith('.') and trace not in ignored:
        print(trace)
        # try min area rect
        input = cv2.imread(PATH_READ +'/' + trace)
        img_copy = cv2.imread(PATH_READ +'/' + trace)
        img_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        # img_blur = cv2.blur(img_gray, (3,3),0)
        img_equ = cv2.equalizeHist(img_gray)
        # print(img_equ.shape)
        ddepth = cv2.CV_8U
        height_img = input.shape[0]

        # img_equ = cv2.bitwise_not(img_equ)
        
        ret, th = cv2.threshold(img_equ,80,255,cv2.THRESH_BINARY_INV)
        img_contours = copy.copy(th)
        contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(input,contours,-1,(0,255,255),2)
        contours_after_size_verification = []
        for contour in contours:
            if verifySize(height_img, contour):
                contours_after_size_verification.append(contour)

        # permanently altering image by drawing contours
        # cv2.drawContours(input,contours_after_size_verification,-1,(0,255,0),3)
        # plt.imshow(input), plt.title("Contours After Size Verification")
        # plt.show()

        for contour in contours_after_size_verification:

            # bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            paddingw = int(w/3)
            paddingh = int(h/4)
            #img_crop used to reference copy of input, and the written file was img crop
            img_crop = cv2.getRectSubPix(img_contours, (w+paddingw,h+paddingh), (x+w/2,y+h/2))
            resized_image = cv2.resize(img_crop,(34,85))
            # resized_image = cv2.resize(img_crop,(224,224))
            # plt.imshow(img_crop)
            # plt.show()
            # cv2.imwrite("{}/characters/test_char".format(path) + str(numImg)+'.png',resized_image)
            cv2.imwrite("{}/test_write/test_char".format(PATH_WRITE) + str(numImg)+'.png',resized_image)

            numImg += 1 
            print("number character saved: ", numImg)
