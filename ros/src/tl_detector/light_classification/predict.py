import os
import numpy as np
import cv2
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.utils import np_utils

model = load_model('inceptionv3-carla.model')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

for root, subdirs, files in os.walk('./dataset'):
    for filename in files:
        file_path = os.path.join(root, filename)
        output_path = './output/' + '/'.join(file_path.strip('/').split('/')[2:])
        img = load_img(file_path, target_size=(299, 299))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x).argmax(axis=-1)[0]
        result = ['unknown', 'red', 'green'][preds]
        print(output_path + '-->' + result)
        img = cv2.imread(file_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,result,(10,10), font, 0.5,(0,0,255),2,cv2.LINE_AA)
        cv2.imwrite(output_path, img)
