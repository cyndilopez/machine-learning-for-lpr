
import os
def move_files():
    IMAGE_PATH = '/Users/cylopez/Documents/projects/license-plate-recognition/plates_mania_lp/california/'
    new_dir = 'openalpr8/'
    os.makedirs(IMAGE_PATH + new_dir)
    ignored = ['openalpr1', 'openalpr2', 'openalpr3', 'openalpr4','openalpr5','openalpr6','openalpr7','openalpr8','extracted', 'Test', 'test_extracted']
    for index, image in enumerate(os.listdir(IMAGE_PATH)):
        if image not in ignored and not image.startswith('.'):
            if index < 1001:
                os.rename(IMAGE_PATH + image, IMAGE_PATH + new_dir + image)
                print( IMAGE_PATH + new_dir + image)

def move_files2():
    IMAGE_PATH = '/Users/cylopez/Documents/projects/license-plate-recognition/plates_mania_lp/california/'
    new_dir = 'not_extracted/'
    exist_dir = 'openalpr7/'
    list_images = []
    for index, image in enumerate(os.listdir(IMAGE_PATH+'extracted')):
        if not image.startswith('.'):
            list_images.append(image[3:-6])

    for index, image in enumerate(os.listdir(IMAGE_PATH+exist_dir)):
        if image[:-6] not in list_images:
            os.rename(IMAGE_PATH + exist_dir + image, IMAGE_PATH + new_dir + image)


def move_files_ocr():
    IMAGE_PATH = '/Users/cylopez/Documents/projects/license-plate-recognition/'
    new_dir = 'ca_pm_characters_recognized/'
    exist_dir = 'ca_pm_characters/characters_detected'
    check_dir = '/Users/cylopez/Documents/projects/license-plate-recognition/ca_pm_characters/'

    list_images = []
    for index, image in enumerate(os.listdir(IMAGE_PATH+exist_dir)):
        if not image.startswith('.'):
            list_images.append(image[3:-4])
            print(image[3:-4])
    for index, image in enumerate(os.listdir(check_dir)):
        print(image[3:10])
        if image[3:10] in list_images:
            print(image)
            os.rename(check_dir + image, IMAGE_PATH + new_dir + image)   

def delete_files():
    IMAGE_PATH = '/Users/cylopez/Documents/projects/license-plate-recognition/plates_mania_lp/california/extracted'
    IMAGE_PATH_COMP = '/Users/cylopez/Documents/projects/license-plate-recognition/plates_mania_lp/california/not_extracted/'
    for index, image in enumerate(os.listdir(IMAGE_PATH)):
        for index_comp, image_comp in enumerate(os.listdir(IMAGE_PATH_COMP)):
                if image[3:-6] == image_comp[:-6]:
                    os.remove(IMAGE_PATH_COMP+image_comp)
                    print(image_comp)

def delete_files2():
    IMAGE_PATH = '/Users/cylopez/Documents/projects/license-plate-recognition/plates_mania_lp/california/extracted'
    IMAGE_PATH_COMP = '/Users/cylopez/Documents/projects/license-plate-recognition/plates_mania_lp/california/openalpr6/'
    for index, image in enumerate(os.listdir(IMAGE_PATH)):
        for index_comp, image_comp in enumerate(os.listdir(IMAGE_PATH_COMP)):
                if image[3:-6] == image_comp[:-6]:
                    os.remove(IMAGE_PATH_COMP+image_comp)
                    print(image_comp)

def delete_files3():
    IMAGE_PATH = '/Users/cylopez/Documents/projects/license-plate-recognition/ca_pm_characters/'
    dir1 = 'characters_detected/'
    dir2 = 'characters_not_detected/'

    for index, image in enumerate(os.listdir(IMAGE_PATH + dir1)):
        if "image" in image:
            os.remove(IMAGE_PATH + dir1 + image)
            print(image)

    for index, image in enumerate(os.listdir(IMAGE_PATH + dir2)):
        if "image" in image:
            os.remove(IMAGE_PATH + dir2 + image)
            print(image)


# delete_files2()
# move_files2()
# delete_files3()
move_files_ocr()