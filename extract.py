import cv2
import random
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
################################# image processing for license plate detection
class Point:
    def __init__(self,x_init,y_init):
        self.x = x_init
        self.y = y_init

def verifySize(candidateContour):
    error = 0.4
    # US license plate size: 12" x 6", aspect 2
    aspect = 2
    # Set a min and max area. All other patches are discarded
    min = 20*aspect*20
    max = 140*aspect*140


    rmin = aspect - aspect*error
    rmax = aspect + aspect*error

    x, y, w, h = cv2.boundingRect(candidateContour)

    area = cv2.contourArea(candidateContour)
    ((x2, y2), (w2, h2), angle) = cv2.minAreaRect(candidateContour)
    r = w/h


    if area < min or area > max or r < rmin or r > rmax:
        return False
    else:
        # print(w*h)
        # print(area)
        # print("Rect: ")
        # print(cv2.minAreaRect(candidateContour))
        # print("Box points: ")
        # print(cv2.boxPoints(cv2.minAreaRect(candidateContour))[0])
        # print("Box int0: ")
        # print(np.int0(cv2.boxPoints(cv2.minAreaRect(candidateContour))))
        return True

ap=argparse.ArgumentParser() # holds all the info necessary to parse the command line into Python data types
ap.add_argument("-i", "--image", type=str, required=True, help="path to image")
args = (ap.parse_args())

# print(args.image)

img = cv2.imread(args.image)
img_copy = cv2.imread(args.image)

# print(img.shape)
plt.imshow(img)
plt.show()

# Convert image to gray
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply a Gaussian blur of 5x5 and remove noise; if noise
# is not removed then we get a lot of vertical edges that
# produce a failed detection
# blur does this by taking the average of all the pixels under a kernel
# area; so here it takes an average of all the pixels under kernel area
# and replaces the central element
img_blur = cv2.blur(img_gray, (5,5),0)

plt.subplot(131), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(img_gray), plt.title('Gray')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(img_blur), plt.title('Blur')
plt.xticks([]), plt.yticks([])
plt.show()

# to find vertical edges, use Sobel filter to find the first
# horizontal derivative. The derivative allows us to find the
# vertical edges on an image
ddepth = cv2.CV_16S
ddepth = cv2.CV_8U

ksize = 5
sobelx = cv2.Sobel(img_blur, ddepth, 1, 0, ksize)
sobely = cv2.Sobel(img_blur, ddepth, 1, 0, ksize)

plt.subplot(221), plt.imshow(sobelx), plt.title('Sobel-horiz k=3')
plt.subplot(222), plt.imshow(sobely), plt.title('Sobel-vert k=3')

ksize = 3
sobelx5 = cv2.Sobel(img_blur, ddepth, 1, 0, ksize)
sobely5 = cv2.Sobel(img_blur, ddepth, 1, 0, ksize)

plt.subplot(223), plt.imshow(sobelx5), plt.title('Sobel-horiz k=5')
plt.subplot(224), plt.imshow(sobely5), plt.title('Sobel-vert k=5')

plt.show()

ret, th = cv2.threshold(sobelx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

plt.imshow(th)
plt.show()

# manually create structuring element, pass shape and size of kernel
element = cv2.getStructuringElement(cv2.MORPH_RECT,(12,3))
# removes blank spaces inside the structuring element
# morph close is dilation followed by erosion, so 
# it increases the size of our object and then removes noise
# it's useful in closing small holes inside foreground objects
th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, element)

plt.imshow(th), plt.title('Remove blank spaces between structural elements')
plt.show()
# three arguments, first is source image, second is contour retrieval mode, third is contour approximation
# outputs contours and hierarchy. each individual contour is a numpy array of (x,y) coordinates of boundary
# points of the object
# curve joining all continuous points along boundary having same color or intensity
contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


# print("Number of Contours found = " + str(len(contours))) 


contours_after_size_verification = []
for contour in contours:
    if verifySize(contour):
        contours_after_size_verification.append(contour)

# print(len(contours_after_size_verification))
# permanently altering image by drawing  contours
cv2.drawContours(img,contours_after_size_verification,-1,(0,255,0),3)
plt.imshow(img), plt.title("Contours After Size Verification")
plt.show()

for contour in contours_after_size_verification:
    x, y, w, h = cv2.boundingRect(contour)
    # permanently altering image by drawing rectangle
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    plt.imshow(img), plt.title("Bounding Box Contours")
    plt.show()

    # bounding rectangle is drawn with minimum area so it also considers rotation
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # permanently altering image by drawing box
    cv2.drawContours(img,[box],0,(0,0,255),2)
    plt.imshow(img), plt.title("Min Rect Contours")
    plt.show()

# all plates have same background color; use a flood fill algorithm
# to retrieve the rotated rectangle for precise cropping
# floodfill fills a connected component with given color
print(len(contours_after_size_verification))
contours_after_ff = []
for contour in contours_after_size_verification:
    ((x2, y2), (w2, h2), angle) = cv2.minAreaRect(contour)
    center = (int(x2), int(y2))

    radius = int(3)
    cv2.circle(img,center,radius,(0,255,255),-1)
    plt.imshow(img)
    plt.show()

    # get min size between width and height
    if w2 < h2:
        minsize = w2
    else:
        minsize = h2

    minsize = minsize - minsize*0.5

    # initialize floodfill parameters and variables    
    # mask is a single-channel 8bit image, 2px wider and 2px taller than image
    mask = np.zeros((img.shape[0] + 2,img.shape[1] + 2), np.uint8)
    lodiff = (30,) #maximal lower brightness/color difference
    updiff = (30,) #maximal upper brightness/color difference
    connectivity = 4
    newmaskval = 255 # new color to put into image
    numseeds = 10
    flags = connectivity | newmaskval << 8   # bit shift
    flags |= cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY

    for j in range(numseeds):
        seedx = x2+random.randint(0, int(minsize - (minsize/2))) 
        seedy = y2+random.randint(0, int(minsize - (minsize/2)))

        # draws a circle
        cv2.circle(img,(int(seedx),int(seedy)),1,(0,255,255),-1)
        
        # fills a connected component with color into a mask image
        # starting from a seed point
        area = cv2.floodFill(img,mask,(int(seedx),int(seedy)),newmaskval,lodiff,updiff,flags)
    print(mask)
    pointsInterest = []
    for i in range(0,len(mask)):
        for j in range(len(mask[0])):
            if mask[i,j] == 255:
                print("found one")
                pointsInterest.append([i,j])
                minRect = cv2.minAreaRect(np.array(pointsInterest))
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # permanently altering image by drawing box
                cv2.drawContours(img,[box],0,(0,0,255),2)
                # plt.imshow(img), plt.title("Min Rect Contours")
                # plt.show()
                contours_after_ff.append(contour)
                # print(minRect)

# cv2.drawContours(img_copy,contours_after_size_verification,-1,(0,255,0),3)
# plt.imshow(img_copy), plt.title("Contours After FF")
# plt.show()

# print(contour)
# print(mask)
# print(area)
# # ## crop detected region, remove rotation, crop image region, resize image, and equalize light of cropped image regions
# # plt.imshow(img_copy)
# # plt.show()
# # for contour in contours_after_size_verification:
# #     # crop image
# #     rect = cv2.minAreaRect(contour)
# #     box = cv2.boxPoints(rect)
# #     box = np.int0(box)

# #     width = int(rect[1][0])
# #     height = int(rect[1][1])
# #     src_pts = box.astype("float32")
# #     dst_pts = np.array([[0, height-1],
# #     [0,0],
# #     [width-1, 0],
# #     [width-1, height-1]], dtype="float32")

# #     M = cv2.getPerspectiveTransform(src_pts, dst_pts)
# #     plt.figure()
# #     warped = cv2.warpPerspective(img_copy, M, (width, height))
# #     plt.imshow(warped)
# #     plt.show()
# #     resized_image = cv2.resize(warped, (150,75))
# #     # apply light histogram equalization since could be different light conditions
# #     img_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
# #     img_blur = cv2.blur(img_gray, (5,5),0)
# #     img_equ = cv2.equalizeHist(img_blur)
# #     # check to see if if detected objects in extract.py can look like this
# #     # also need to save some objects that are NOT license plates
# #     # print(resized_image.shape)
# #     print(img.shape)
# #     plt.imshow(img_equ)
# #     plt.show()

## crop detected region, remove rotation, crop image region, resize image, and equalize light of cropped image regions
numImg = 493
for contour in contours_after_size_verification:

    # bounding rect
    x, y, w, h = cv2.boundingRect(contour)
 
    # crop image
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    center = rect[0]
    width = int(rect[1][0])
    height = int(rect[1][1])
    angle = int(rect[2])

    # Get rotation matrix
    r_asp = width/height

    if r_asp < 1:
        angle = 90 + angle
    
    # generates transform matrix
    rotmat = cv2.getRotationMatrix2D(center, angle, 1)
    # warpAffine takes a 2x3 transformation matrix
    img_rot = cv2.warpAffine(img_copy, rotmat, (img_copy.shape[0],img_copy.shape[1]))

    if r_asp < 1:
        temp = height
        height = width
        width = temp
    
    img_crop = cv2.getRectSubPix(img_copy, (w,h), (x+w/2,y+h/2))
    plt.imshow(img_crop)
    plt.show()

#     # cv2.imwrite("./test_data/test" + str(numImg)+'.png',img_crop)
#     # cv2.imwrite("./us_not_license_plates/nonLP" + str(numImg)+'.png',img_crop)

#     numImg += 1

