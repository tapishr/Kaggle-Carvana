from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Dense, Flatten
from keras.optimizers import RMSprop
from keras.applications.vgg16 import VGG16

from losses import bce_dice_loss, dice_loss, dice_coeff


def get_vgg16(input_shape=(128, 128, 3),
                num_classes=1):
"""
Compile a fully convolutional VGG-16 model.
Parameters:
input_shape -- shape of model input
num_classes -- number of segmentation classes
"""
    # Initialize the convolutional layers of VGG-16
    model_top = VGG16(include_top=False, weights=None, input_shape=(128,128,3))

    # Flatten the last convolutional layer to serve as input for the next layer
    last_conv = Flatten()(model_top.layers[-1].output)

    # Initialize the dense layers as convolutional layer
    fc1 = Dense(4096, activation='relu')(last_conv)
    fc2 = Dense(4096, activation='relu')(fc1)
    output = Dense(num_classes*128*128, activation='softmax')(fc2)

    # Combine the layers to form a fully convolutional model
    model = Model(inputs=model_top.input, outputs=output)

    # Compile the model
    model.compile(optimizer=RMSprop(lr=0.0001), loss=bce_dice_loss, metrics=[dice_coeff])

    return model
