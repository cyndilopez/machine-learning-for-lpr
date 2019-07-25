import os
import cv2

def crop_image(pathname):
    img = cv2.imread(pathname)
    width = img.shape[1]
    height = img.shape[0]
    crop_img = img[int(height/12):height-int(height/12), 0:width]
    return crop_img

list_folders_init = ['usimages/characters_detected_ca', 'usimages/characters_detected_wa','usimages/characters_detected_la', 'usimages/characters_detected_az', 'usimages/final_initial_characters_to_test']
exp_transflearn_dataset_path = 'expanded_transflearn_database/'

list_folders = ['/Users/cylopez/Documents/projects/license-plate-recognition/scraped_plates_ca/characters/characters_resized/characters_detected_ca']
states = ['al', 'ar', 'co', 'ct', 'dc', 'de', 'fl', 'ga', 'hi', 'ia', 'id', 'il', 'in']

list_folders = []
for state in states:
    direc = 'scraped_plates_' + state + '/characters/characters_detected_' + state
    list_folders.append(direc)

print(list_folders)
for i in list_folders_init:
    list_folders.append(i)


# exp_transflearn_dataset_path = '/Users/cylopez/Documents/projects/license-plate-recognition/scraped_plates_ca/characters/characters_resized/characters_detected_ca/transflearn/'
# init_transflearn_dataset_path = '/Users/cylopez/Documents/projects/license-plate-recognition/usimages/transflearn'

# class_list = [trace for trace in os.listdir(init_transflearn_dataset_path) if not trace.startswith('.')]

folder_dict = {'A':'A', 'B':'B', 'C':'C', 'D':'D', 
               'E':'E', 'F':'F', 'G':'G', 'H':'H',
               'I':'I', 'J':'J', 'K':'K', 'L':'L',
               'M':'M', 'N':'N', 'O':'O', 'P':'P',
               'Q':'Q', 'R':'R', 'S':'S', 'T':'T',
               'U':'U', 'V':'V', 'W':'W', 'X':'X',
               'Y':'Y', 'Z':'Z', '0':'zero', '1':'one', '2':'two',
               '3':'three', '4':'four', '5':'five','6':'six',
               '7':'seven', '8':'eight', '9':'nine'}
numImg = 0
ignored = ['Not correct']

for dir in list_folders:
    for image in os.listdir(dir):
        if not image.startswith('.') and image not in ignored:
            # cropped_img = crop_image(dir + '/' + image)
            cropped_img = cv2.imread(dir + '/' + image)

            character = image[1]

            # make directory for character if it doesn't already exist
            folder = folder_dict.get(character)
            try:
                os.makedirs(exp_transflearn_dataset_path + folder_dict.get(character))
            except:
                print('')
            
            # os.rename(dir + '/' + inner_folder + '/' + image, exp_transflearn_dataset_path + '/' + folder + '/' + image)
            try: 
                cv2.imwrite(exp_transflearn_dataset_path + folder + '/l' + character + '_' + str(numImg)+'.png',cropped_img)
            except:
                print(character, ' ', folder, ' ', dir + '/' + image)
            
            numImg += 1
