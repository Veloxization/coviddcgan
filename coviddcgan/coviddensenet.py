from keras.preprocessing.image import ImageDataGenerator
from keras.applications import densenet
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Reshape, Concatenate
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers
from keras import backend as K

import numpy as np

K.set_learning_phase(1)
img_width, img_height = 128, 128
nb_train_samples = 40000 # Change this between augmented and non-augmented datasets
nb_validation_samples = 4535
epochs = 10
batch_size = 32
n_classes = 2

def build_model():
    base_model = densenet.DenseNet121(input_shape=(img_width, img_height, 3),
                                      weights="imagenet",
                                      include_top=False,
                                      pooling='avg')
    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = Dense(1000, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = Dense(500, kernel_regularizer=regularizers.l1_l2(0.01), activity_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    created_model = Model(inputs=base_model.input, outputs=predictions)
    
    return created_model

# Build and compile the DenseNet model
model = build_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])

early_stop = EarlyStopping(monitor='loss', patience=8, verbose=1, min_delta=1e-4)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
callbacks_list = [early_stop, reduce_lr]

dataset = np.load('augmented_dataset.npz') # Change this to reflect the dataset to train on
x_train, y_train = dataset['x_train'], dataset['y_train']
x_test, y_test = dataset['x_test'], dataset['y_test']

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Expand dimensions for grayscale images
x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]
x_train = np.concatenate([x_train, x_train, x_train], axis=-1)
x_test = np.concatenate([x_test, x_test, x_test], axis=-1)

# Train model
model.fit(
    x_train, y_train,
    epochs=epochs,
    validation_data=(x_test, y_test),
    validation_steps=nb_validation_samples // batch_size,
    callbacks=callbacks_list)

model_path = "saved_densenet/densenet.json"
weights_path = "saved_densenet/densenet_weights.hdf5"
options = {"file_arch": model_path,
           "file_weight": weights_path}
json_string = model.to_json()
open(options['file_arch'], 'w').write(json_string)
model.save_weights(options['file_weight'])

# Evaluate the model on the test data
test_result = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest results (loss, acc, mse): {test_result}')
