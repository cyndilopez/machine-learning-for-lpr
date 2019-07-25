from keras.models import load_model

from keras.preprocessing import image
import os
import numpy as np

PATH = 'usimages/testdir'
model_filename = 'checkpoints/RN50-weights.02-2.85.hdf5'
categories = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 
              'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 
              'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 
              'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 
              'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 
              'Z': 25, 'eight': 26, 'five': 27, 'four': 28, 'nine': 29, 
              'one': 30, 'seven': 31, 'six': 32, 'three': 33, 'two': 34}
image_width = 224
image_height = 224

model = load_model(model_filename)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

for jj, folder in enumerate(os.listdir(PATH)):
    if not folder.startswith('.'):
        folder_path = PATH + '/' + folder
        for index, image_path in enumerate(os.listdir(folder_path)):
            if not image_path.startswith('.'):
                test_image = image.load_img(folder_path+'/'+image_path, target_size = (image_height, image_width)) 
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis = 0)
                #predict the result
                result = model.predict(test_image)
                predictions = np.argmax(result, axis=1) #in an array
                print("Actual: ", folder)
                print("Predicted: ", list(categories.keys())[list(categories.values()).index(predictions[0])])
