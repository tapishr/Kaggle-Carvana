from keras.losses import binary_crossentropy
import keras.backend as backend


def dice_coeff(y_true, y_pred):
"""
Computes the dice coefficient.
Parameters:
y_true -- ground truth
y_pred -- model output

"""
    smooth = 1.
    y_true_f = backend.flatten(y_true)
    y_pred_f = backend.flatten(y_pred)
    intersection = backend.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (backend.sum(y_true_f) + backend.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
"""
Computes the dice loss.
Parameters:
y_true -- ground truth
y_pred -- model output

"""
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
"""
Combine the binary crossentropy and dice loss.
Parameters:
y_true -- ground truth
y_pred -- model

"""
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss
