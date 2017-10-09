import os
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from styx_msgs.msg import TrafficLight
from light_classification.train import IMAGE_WIDTH, IMAGE_HEIGHT

default_model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inceptionv3-carla.model')

class TLClassifierDL(object):
    def __init__(self, path=default_model_file):
        model = load_model(path)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.model = model
        self.graph = tf.get_default_graph()

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        
        """
        preprocessed = self.preprocess(image)
        prediction = self.predict(preprocessed)

        print(prediction)

        prediction = [TrafficLight.UNKNOWN, TrafficLight.RED, TrafficLight.GREEN][prediction]

        return TrafficLight.RED if prediction == TrafficLight.RED else TrafficLight.UNKNOWN

    def preprocess(self, input_img):
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
        input_img = img_to_array(input_img)
        input_img = np.expand_dims(input_img, axis=0)
        input_img = preprocess_input(input_img)
        return input_img

    def predict(self, image):
        with self.graph.as_default():
            prediction = self.model.predict(image).argmax(axis=-1)[0]
        return prediction
