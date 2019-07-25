import argparse
import requests
import cv2
import os
from imutils import paths
#  python scrape_google_images.py --urls urls_al.txt -o scraped_plates_al

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True,
                help="path to file containing image URLs")
ap.add_argument("-o","--output", required=True,
                help="path to output directory of images")
args = vars(ap.parse_args())

# os.makedirs('scraped_plates_{}'.format(state))


# grab the list of URLs from the input file, then initialize the 
# total number of images downloaded thus far
rows = open(args["urls"]).read().strip().split("\n")
total = 0

for url in rows:
    try:
        # try to download the image
        r = requests.get(url,timeout=60)

        # save the image to disk
        p = os.path.sep.join([args["output"], "{}.jpg".format(
			str(total).zfill(8))])
        f = open(p, "wb")
        f.write(r.content)
        f.close()

        # update counter
        print("[INFO] downloaded: {}".format(p))
        total += 1
    except:
        print("[INFO] error downloading {}...skipping")

for imagePath in paths.list_images(args["output"]):
    delete = False
    try:
        image = cv2.imread(imagePath)
        if image is None:
            delete = True

    except:
        print("Except")
        delete = True
    
    if delete: 
        os.remove(imagePath)