##################    # for testing purposes, segmentation and character detection on one image that's fed in the command line

import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
import os
from send_GCR import detect_text

def verifySize(contour):
    # return True
    # char sizes are 45x77
    # char sizes are 2.5"x2.5*(3 to 4)
    x, y, w, h = cv2.boundingRect(contour)


    aspect = 2.5/(2.5*2.5) #based on eyeballing
    charAspect = w/h
    error = 0.6
    # for image shape of (164, 287)
    minHeight = 30
    minAspect = 0.2
    maxAspect = aspect+aspect*error

    if charAspect > minAspect and charAspect < maxAspect and h>=minHeight:
        # permanently altering image by drawing rectangle
        cv2.rectangle(input,(x,y),(x+w,y+h),(0,0,255),2)
        plt.imshow(th), plt.title("Bounding Box Contours")
        plt.show()
        return True
    else:
        return False

ap=argparse.ArgumentParser() # holds all the info necessary to parse the command line into Python data types
ap.add_argument("-i", "--image", type=str, required=True, help="path to image")
args = (ap.parse_args())

input = cv2.imread(args.image)
img_copy = cv2.imread(args.image)
img_gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
img_blur = cv2.blur(img_gray, (3,3),0)
img_equ = cv2.equalizeHist(img_blur)
# print(img_equ.shape)
ddepth = cv2.CV_8U

ret, th = cv2.threshold(img_equ,60,255,cv2.THRESH_BINARY_INV)
img_contours = copy.copy(th)
contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# cv2.drawContours(input,contours,-1,(0,255,255),2)
contours_after_size_verification = []
for contour in contours:
    if verifySize(contour):
        contours_after_size_verification.append(contour)

# permanently altering image by drawing  contours
cv2.drawContours(input,contours_after_size_verification,-1,(255,255,0),3)
plt.imshow(input), plt.title("Contours After Size Verification")
plt.show()
numImg = 0
for contour in contours_after_size_verification:

    # bounding rect
    x, y, w, h = cv2.boundingRect(contour)
    paddingw = int(w/5)
    paddingh = int(h/5)
    # print(paddingw)
    # print(paddingh)
    img_crop = cv2.getRectSubPix(img_contours, (w+paddingw,h+paddingh), (x+w/2,y+h/2))
    resized_image = cv2.resize(img_crop,(200,200))
    print(resized_image[20,:])
    plt.imshow(resized_image)
    plt.show()
    # cv2.imwrite("./usimages/characters/ca_char" + str(numImg)+'.png',img_crop)
    # detect_text("./usimages/characters/ca_char" + str(numImg)+'.png')
    numImg += 1 
    # print("number character saved: ", numImg)

