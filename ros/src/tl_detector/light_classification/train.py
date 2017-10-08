import os
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

IMAGE_WIDTH, IMAGE_HEIGHT = 299, 299 #fixed size for InceptionV3
EPOCH = 5
TOTAL_SAMPLES = sum([len(files) for r, d, files in os.walk('./dataset')])
VAL_SAMPLES = 4
BATCH_SIZE = 100

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

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        './dataset',
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE)

    validation_generator = test_datagen.flow_from_directory(
        './dataset',
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE)

    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer 4 classes
    # Red, Yellow, Green, Unknown
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # Transfer Learning
    # model.fit_generator(
    #     train_generator,
    #     nb_epoch=EPOCH,
    #     samples_per_epoch=TOTAL_SAMPLES,
    #     validation_data=validation_generator,
    #     nb_val_samples=VAL_SAMPLES,
    #     class_weight='auto')
    model.fit_generator(
        train_generator,
        steps_per_epoch=TOTAL_SAMPLES//BATCH_SIZE,
        epochs=EPOCH,
        class_weight='auto',
        shuffle=True)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    print(model.summary())
    # Fine Tuning
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    model.fit_generator(
        train_generator,
        steps_per_epoch=TOTAL_SAMPLES//BATCH_SIZE,
        class_weight='auto',
        shuffle=True)

    # model.fit_generator(
    #     train_generator,
    #     nb_epoch=EPOCH,
    #     samples_per_epoch=TOTAL_SAMPLES,
    #     validation_data=validation_generator,
    #     nb_val_samples=VAL_SAMPLES,
    #     class_weight='auto')

    model.save('inceptionv3-carla.model')

if __name__=="__main__":
    create_model()
