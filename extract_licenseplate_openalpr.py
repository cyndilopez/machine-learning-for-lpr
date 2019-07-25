import requests
import base64
import json
import cv2
import os

def return_data_openalpr(IMAGE_PATH):

    with open(IMAGE_PATH, 'rb') as image_file:
        img_base64 = base64.b64encode(image_file.read())

    url = 'https://api.openalpr.com/v2/recognize_bytes?recognize_vehicle=1&country=us&secret_key=%s' % (SECRET_KEY)
    r = requests.post(url, data = img_base64)

    # print(json.dumps(r.json(), indent=2))
    data = json.loads(r.text)
    if data["results"] != []:
        data_des = clean_data_results(data["results"][0])
        return data_des
    else:
        return None

def clean_data_results(data):
        des_data = {
            'coordinates': data["coordinates"],
            'state': data["region"],
            'plate': data["plate"]
        }
        return des_data

def get_coord(data_des):
        min_xcoord = min(pair["x"] for pair in data_des["coordinates"])
        min_ycoord = min(pair["y"] for pair in data_des["coordinates"])
        max_xcoord = max(pair["x"] for pair in data_des["coordinates"])
        max_ycoord = max(pair["y"] for pair in data_des["coordinates"])
        return min_xcoord, min_ycoord, max_xcoord, max_ycoord

IMAGE_PATH = '/Users/cylopez/Documents/projects/license-plate-recognition/plates_mania_lp/california/'
openalpr_dir = 'not_extracted/'
FULL_IMAGE_PATH = IMAGE_PATH + openalpr_dir
for index,image_file in enumerate(os.listdir(FULL_IMAGE_PATH)):
    if not image_file.startswith('.'):
        # api request
        data_des = return_data_openalpr(IMAGE_PATH + openalpr_dir + image_file)
        if data_des:
            # get coordinates for individual license plate
            min_xcoord, min_ycoord, max_xcoord, max_ycoord = get_coord(data_des)  
            # read image
            img = cv2.imread(IMAGE_PATH + openalpr_dir + image_file)
            # crop
            img_crop = img[int(min_ycoord):int(max_ycoord),int(min_xcoord):int(max_xcoord)]
            # save file
            cv2.imwrite(IMAGE_PATH + 'extracted/lp_' + image_file, img_crop)
