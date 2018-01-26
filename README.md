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
### Licence
MIT License

Copyright (c) [2014] [Tapish Rathore]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
