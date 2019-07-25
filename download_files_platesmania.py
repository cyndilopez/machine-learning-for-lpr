import requests
import xml.etree.ElementTree as ET

# structure is as follows:
## plate
### car
### model
### photo_url
### plate_number
### plate_number_image_url
### plate_type
### plate_type_title
### plate_regions
### link

PATH ='/Users/cylopez/Documents/projects/license-plate-recognition/plates_mania_lp/california'
f = ET.parse("50k_us.xml")
root = f.getroot()

print(root[0][0].text)
print(root[0][1].text)
print(root[0][2].text)
print(type(str(root[0][3].text)))
print(root[0][4].text)
print(root[0][5].text)
print(root[0][6].text)
print(root[0][7].text)
print(root[0][8].text)
count = 0
for child in root:
    plate_region = child[7].text
    plate_name = child[3].text
    photo_url = child[2].text
    if plate_region == "California":
        if len(plate_name)==7:
            if " " not in plate_name:
                # all of these conditions give 11,805 images
                count += 1
                response = requests.get(photo_url, stream=True)
                if response.status_code == 200:
                    with open(PATH + '/' + plate_name + "ca.png", 'wb') as f:
                        for chunk in response:
                            f.write(chunk)

print(count)

