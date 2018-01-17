import cv2
import numpy as np
import pandas as pd
import threading
import Queue
import io
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tqdm import tqdm

import params

rles = []
def test(output_dir, weights_dir, test_dir):
"""
Tests the given model and generates a csv containing segmentation masks compressed using rle
Parameters:
output_dir -- directory path to store output csv
weights_dir -- directory path to load weights of given model
test_dir -- directory path containing test images

"""

    batch_size = params.batch_size
    
    model = params.model_factory()

    # Copy file from gcs to local directory
    with file_io.FileIO(weights_dir, mode='r') as input_f:
        with file_io.FileIO('weights.hdf5', mode='w+') as output_f:
            output_f.write(input_f.read())

    model.load_weights(filepath='weights.hdf5')
    graph = tf.get_default_graph()

    with file_io.FileIO(test_dir + '/sample_submission.csv', mode='r') as f:
        csv_bytes = f.read()
        df_test = pd.read_csv(io.BytesIO(csv_bytes))

    ids_test = df_test['img'].map(lambda s: s.split('.')[0])

    names = []
    for id in ids_test:
        names.append('{}.jpg'.format(id))

    q_size = 10

    q = Queue.Queue(maxsize=q_size)
    t1 = threading.Thread(target=data_loader, name='DataLoader', args=(q, ids_test, test_dir))
    t2 = threading.Thread(target=predictor, name='Predictor', args=(q, len(ids_test), graph, model))
    print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
    t1.start()
    t2.start()
    # Wait for both threads to finish
    t1.join()
    t2.join()

    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})

    # Copy submission file to gcs
    df.to_csv('submission.csv.gz', index=False, compression='gzip')
    with file_io.FileIO('submission.csv.gz', mode='r') as input_f:
        with file_io.FileIO(output_dir + '/submission.csv.gz', mode='w+') as output_f:
            output_f.write(input_f.read())


# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
"""
Encodes given mask using run length encoding

"""
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle


def data_loader(q, ids_test, test_dir):
"""
Loads images, puts them in a batch, and adds them to a queue
"""
    batch_size = params.batch_size
    input_size = params.input_size
    for start in range(0, len(ids_test), batch_size):
        x_batch = []
        end = min(start + batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        for idx in ids_test_batch.values:
            with file_io.FileIO(test_dir + '/{}.jpg'.format(idx), mode='r') as f:
              image_bytes = f.read()
            
            nparr = np.fromstring(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            input_size = params.input_size
            img = cv2.resize(img, (input_size, input_size))
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255
        q.put(x_batch)


def predictor(q, len_ids_test, graph, model):
"""
Predicts segmentation masks of given images
"""
    batch_size = params.batch_size
    orig_width = params.orig_width
    orig_height = params.orig_height
    for i in tqdm(range(0, len_ids_test, batch_size)):
        x_batch = q.get()
        with graph.as_default():
            preds = model.predict_on_batch(x_batch)
        preds = np.squeeze(preds, axis=3)
        for pred in preds:
            prob = cv2.resize(pred, (orig_width, orig_height))
            mask = prob > params.threshold
            rle = run_length_encode(mask)
            rles.append(rle)






if __name__ == '__main__':
    
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--job-dir',
      help='GCS or local paths to test data',
      required=True
    )
    
    parser.add_argument(
      '--weights-dir',
      help='GCS location or local paths to weights',
      required=True
    )

    parser.add_argument(
      '--output-dir',
      help='GCS location to write output',
      required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__
    output_dir = arguments.pop('output_dir')
    weights_dir = arguments.pop('weights_dir')
    test_dir = arguments.pop('job_dir')

    test(output_dir, weights_dir, test_dir)
