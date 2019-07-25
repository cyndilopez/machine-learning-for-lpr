from send_GCR import detect_text
import os 
import cv2
import statistics



# PATH_READ = '/Users/cylopez/Documents/projects/license-plate-recognition/cali_characters/'
PATH_READ = '/Users/cylopez/Documents/projects/license-plate-recognition/plates_mania_lp/test/'
TRACES = os.listdir(PATH_READ)
def save_char_imgs_detected():
    for jj, trace in enumerate(TRACES):
        if "png" in trace:
            full_path = PATH_READ + trace
            print(full_path)
            detect_text(PATH_READ,full_path,jj,trace)

save_char_imgs_detected()


