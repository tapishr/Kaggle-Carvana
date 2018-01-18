# Kaggle-Carvana
This is a submission for the [Carvana Image segmentation challenge](https://www.kaggle.com/c/carvana-image-masking-challenge/) 
on Kaggle. This solution trains U-nets on Google Cloud Platform to segment car images.
## Instructions
To run the code on a local machine, requirements include -
- Python 2.7
- Tensorflow (with Python 2.7 wrapper)
- OpenCV
- Other libraries that can be installed using `setup.py` script provided with the code

There are 2 sets of code – one for training the models and one for testing. 
Both training and testing code have been designed to run on the Google Cloud Platform (gcp).
To run the code on gcp, run the `start.sh` shell script by typing the following command -

`$ sh start.sh`

Values for arguments in `start.sh` would need to be changed before running it -
- Inside start.sh, the argument `job-dir` must contain the GCS (Google Cloud Storage) URL which will store logs and the model’s weight file. 
- The argument `train-dir` should contain the GCS URL where all the training images and the training mask csv file are stored.

To run the code locally, the script `local.sh` can be executed after giving valid directory paths for `job-dir` and `train-dir` arguments.
