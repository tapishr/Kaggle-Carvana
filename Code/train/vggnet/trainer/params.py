from model.vgg_net import get_vgg16

input_size = 128

max_epochs = 100
batch_size = 1

orig_width = 1918
orig_height = 1280

threshold = 0.5

model_factory = get_vgg16
