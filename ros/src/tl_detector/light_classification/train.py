import os
import numpy as np
from numpy import genfromtxt # Read csv
from skimage import io
from sklearn.cross_validation import train_test_split
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.optimizers import SGD
from keras.utils import np_utils

IMAGE_WIDTH, IMAGE_HEIGHT = 299, 299 #fixed size for InceptionV3
NUM_CLASSES = 3
EPOCH = 25
TOTAL_SAMPLES = sum([len(files) for r, d, files in os.walk('./dataset')])
VAL_SAMPLES = 4
BATCH_SIZE = 32

def load_images():
    X = []
    y = []
    dataset = genfromtxt('./dataset.csv', delimiter=',',dtype=None)

    for path, target in dataset:
        img = load_img('./dataset/' + path + '.jpg', target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
        img = img_to_array(img)
        X.append(img)
        y.append(target)

    X = np.array(X)
    y = np_utils.to_categorical(np.array(y), NUM_CLASSES)

    print(X.shape)
    print(y.shape)
    return X, y

def create_model():
    # data prep
    train_datagen =  ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    X, y = load_images()
    X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=42)

    train_datagen.fit(X_train)
    val_datagen.fit(X_val)

    # Using csv
    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    validation_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)


    # train_generator = train_datagen.flow_from_directory(
    #     './dataset',
    #     target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    #     batch_size=BATCH_SIZE)
    #
    # validation_generator = val_datagen.flow_from_directory(
    #     './dataset',
    #     target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    #     batch_size=BATCH_SIZE)

    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer 4 classes
    # Red, Yellow, Green, Unknown
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    print(model.summary())

    # Transfer Learning
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(y_train)//BATCH_SIZE,
        epochs=EPOCH,
        validation_data=validation_generator,
        validation_steps=len(y_val)//BATCH_SIZE,
        class_weight='auto')

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    # Fine Tuning
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    model.fit_generator(
        train_generator,
        steps_per_epoch=len(y_train)//BATCH_SIZE,
        epochs=EPOCH,
        validation_data=validation_generator,
        validation_steps=len(y_val)//BATCH_SIZE,
        class_weight='auto')

    model.save('inceptionv3-carla.model')

if __name__=="__main__":
    create_model()
