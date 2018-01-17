import cv2
import numpy as np
import pandas as pd
import argparse
from tensorflow.python.lib.io import file_io
import io
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.model_selection import train_test_split

class ModelSave(keras.callbacks.Callback):
  """
  Callback methods used during training, specifically for saving model and model weights.
  """
    def __init__(self, filepath, monitor='val_loss',
                 save_best_only=False, save_weights_only=False):
    """
    Initialization function. 
    Parameters:
    filepath -- directory path where model is saved
    monitor -- metric which is observed to determine model performance
    save_best_only -- only saves the model with best performance
    save_weights_only -- only saves weights of the model ignoring model architecture

    """
        super(ModelSave, self).__init__()
        self.monitor = monitor
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.best = np.Inf

    def on_batch_end(self, batch, logs={}):
    """
    Callback function called after processing one batch during training.
    Parameters:
    batch -- batch of images processed
    logs -- training logs of processed batch

    """
        if self.save_best_only :
            current = logs.get(self.monitor)
            if np.less(current, self.best):
                self.best = current
                if self.save_weights_only:
                  # First save the model weights in local machine (on google compute engine)
                    self.model.save_weights('vggnet_best_weights.hdf5', overwrite=True)
                  # Now copy the weights file to google compute storage path dtored in self.filepath 
                    with file_io.FileIO('vggnet_best_weights.hdf5', mode='r') as input_f:
                        with file_io.FileIO(self.filepath + '/vggnet_best_weights.hdf5', mode='w+') as output_f:
                            output_f.write(input_f.read())
                else:
                  # First save the model in local machine (on google compute engine)
                    self.model.save('vggnet_best_model.h5', overwrite=True)
                  # Now copy the file to google compute storage path dtored in self.filepath 
                    with file_io.FileIO('vggnet_best_model.h5', mode='r') as input_f:
                        with file_io.FileIO(self.filepath + '/vggnet_best_model.h5', mode='w+') as output_f:
                            output_f.write(input_f.read())
        else:
            if self.save_weights_only:
              # First save the model weights in local machine (on google compute engine)
                self.model.save_weights('vggnet_weights.hdf5', overwrite=True)
              # Now copy the weights file to google compute storage path dtored in self.filepath 
                with file_io.FileIO('vggnet_weights.hdf5', mode='r') as input_f:
                    with file_io.FileIO(self.filepath + '/vggnet_weights.hdf5', mode='w+') as output_f:
                        output_f.write(input_f.read())
            else:
              # First save the model in local machine (on google compute engine)
                self.model.save('vggnet_model.h5', overwrite=True)
              # Now copy the file to google compute storage path dtored in self.filepath 
                with file_io.FileIO('vggnet_model.h5', mode='r') as input_f:
                    with file_io.FileIO(self.filepath + '/vggnet_model.h5', mode='w+') as output_f:
                        output_f.write(input_f.read())


def train_vggnet(train_dir, job_dir):
  """
  Trains a fully convolutional VGG-16 CNN model.
  Parameters:
  train_dir -- directory containing training data
  job_dir -- directory where output files (log and weight files) are stored

  """
  # Get parameteers from params file
    import params

    input_size = params.input_size
    epochs = params.max_epochs
    batch_size = params.batch_size
    model = params.model_factory()

  # Get image names from csv file
    with file_io.FileIO(train_dir + '/train_masks.csv', mode='r') as f:
        csv_bytes = f.read()
        df_train = pd.read_csv(io.BytesIO(csv_bytes))
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])

  # Split images into training and validation sets
    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

    print('Training on {} samples'.format(len(ids_train_split)))
    print('Validating on {} samples'.format(len(ids_valid_split)))

  # Initialize callbacks
    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=3,
                               verbose=1,
                               min_delta=1e-4),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=4,
                                   verbose=1,
                                   epsilon=1e-4),
                 ModelSave(monitor='val_loss',
                           filepath=job_dir,
                           save_best_only=True,
                           save_weights_only=True),
                 TensorBoard(log_dir=job_dir + '/logs')]
  
  # Start training
    model.fit_generator(generator=train_generator(train_dir, ids_train_split, batch_size, input_size),
                        steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                        epochs=epochs,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=valid_generator(train_dir, ids_valid_split, batch_size, input_size),
                        validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
    



def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
"""
Shifts the hsv values of the image by a random amount.
Parameters:
image -- image whose values are to be shifted
hue_shift_limit = tuple containing max and min shift applied on hue channel
sat_shift_limit = tuple containing max and min shift applied on saturation channel
val_shift_limit = tuple containing max and min shift applied on value channel
u -- probability of applying all these shifts

"""
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
"""
Shifts, scales and rotates the given image and its mask
Parameters:
image -- image to be transformed
mask -- mask of the corresponding image
shift_limit -- tuple of min and max shift limits
scale_limit -- tuple of min and max scaling limits
rotate_limit -- tuple of min and max rotating limits
borderMode -- color of border of transformed image
u -- probability of applying all these transforms

"""
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
"""
Flips the image horizontally
Parameters:
image -- image to be flipped
mask -- mask of the corresponding image
u -- probability of flipping image

"""
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def train_generator(train_dir, ids_train_split, batch_size, input_size):
"""
Generator function for producing batches of pre processed images and masks.
Parameters:
train_dir -- directory containing training data
ids_train_split -- names of images used for training
batch_size -- number of images in individual batch
input_size -- size of image/first layer of CNN model

"""
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for idx in ids_train_batch.values:
              
                # Read image file  
                with file_io.FileIO(train_dir + '/train/{}.jpg'.format(idx), mode='r') as f:
                  image_bytes = f.read()
                
                nparr = np.fromstring(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Reshape image to fit input layer of CNN model
                img = cv2.resize(img, (input_size, input_size))
                
                # Read mask file  
                with file_io.FileIO(train_dir + '/train_masks/{}_mask.png'.format(idx), mode='r') as f:
                  mask_bytes = f.read()

                nparr = np.fromstring(mask_bytes, np.uint8)
                mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

                # Reshape mask to fit output layer of CNN model
                mask = cv2.resize(mask, (input_size, input_size))
                
                # Apply pre processing
                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                img, mask = randomShiftScaleRotate(img, mask,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-0, 0))
                img, mask = randomHorizontalFlip(img, mask)

                # Flatten mask to fit output layer
                mask = mask.flatten()
                
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


def valid_generator(train_dir, ids_valid_split, batch_size, input_size):
"""
Generator function for producing batches of pre processed validation images and masks.
Parameters:
train_dir -- directory containing training data
ids_valid_split -- names of images used for validation
batch_size -- number of images in individual batch
input_size -- size of image/first layer of CNN model

"""
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for idx in ids_valid_batch.values:

                # Read image file  
                with file_io.FileIO(train_dir + '/train/{}.jpg'.format(idx), mode='r') as f:
                  image_bytes = f.read()
                
                nparr = np.fromstring(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Reshape image to fit input layer of CNN model
                img = cv2.resize(img, (input_size, input_size))
                
                # Read mask file  
                with file_io.FileIO(train_dir + '/train_masks/{}_mask.png'.format(idx), mode='r') as f:
                  mask_bytes = f.read()

                nparr = np.fromstring(mask_bytes, np.uint8)
                mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

                # Reshape mask to fit output layer of CNN model
                mask = cv2.resize(mask, (input_size, input_size))
                # Flatten mask to fit output layer
                mask = mask.flatten()

                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--train-file',
      help='GCS or local paths to training data',
      required=True
    )

    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__

    # Get directory paths from input arguments
    job_dir = arguments.pop('job_dir')
    train_dir = arguments.pop('train_file')
    
    train_vggnet(train_dir, job_dir)


