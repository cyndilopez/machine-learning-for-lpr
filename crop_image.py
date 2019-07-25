import cv2
import os
pathname = "usimages/final_initial_characters_to_test/"
pathname = 'scraped_plates_wa/characters'
for index, image_path in enumerate(os.listdir(pathname)):
    
    if not image_path.startswith('.'):
        img = cv2.imread(pathname+'/'+image_path)
        width = img.shape[1]
        height = img.shape[0]
        print(img.shape)
        crop_img = img[int(height/12):height-int(height/12), 0:width]

        # crop_img = img[y:y+h, x:x+w]
        cv2.imshow("cropped", crop_img)
        cv2.waitKey(0) 

