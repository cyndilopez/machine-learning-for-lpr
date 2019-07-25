###################### uses Google Character Recognition to detect character in segmented license plate image
####################### saves image to new directory with renamed filename to be used for machine learning
def filter_text(texts, path, full_path, index, image):
    import cv2
    import matplotlib.pyplot as plt
    img = cv2.imread(full_path)

    # save both character read and image filename
    num_char_found = 0
    invalid_chars = ['-','?', '.', "'",'"','*','+']
    valid_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                    'W', 'X', 'Y', 'Z']
    if texts == None:
        print("0 characters found: ")
        print(path)
        cv2.imwrite('{}characters_not_detected/l'.format(path)+'_'+image[3:-4]+'.png',img)
    else: 
        for text in texts:
            if text in valid_chars:
                num_char_found += 1
                cv2.imwrite('{}characters_detected/l'.format(path)+text+'_'+image[3:-4]+'.png',img)
                break
    
    if texts != None and num_char_found == 0:
        print("Only invalid characters found: ")
        print(path)
        cv2.imwrite('{}characters_not_detected/l'.format(path)+texts+'_'+image[3:-4]+'.png',img)

def detect_text(path_to_pass,path,index,imagefile):
    import io
    import os
    # Imports the Google Cloud client library
    from google.cloud import vision
    from google.cloud.vision import types


    # Instantiates a client
    client = vision.ImageAnnotatorClient()

    # The name of the image file to annotate
    file_name = os.path.join(
        os.path.dirname(__file__),
        path)

    # Loads the image into memory
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('\nTexts:')
    # print(texts[0].description)
    print(bool(texts))
    if texts:
        text = texts[0].description
    else:
        text = None
    # for text in texts:
        # print('\n"{}"'.format(text.description))

        # vertices = (['({},{})'.format(vertex.x, vertex.y)
        #             for vertex in text.bounding_poly.vertices])

        # print('bounds: {}'.format(','.join(vertices)))
        # print("text: ", text.description)
    filter_text(text, path_to_pass, path, index, imagefile)
