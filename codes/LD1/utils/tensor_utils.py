"""
Utility functions for tensor related operations
"""

import logging
import os

import numpy as np
import scipy.io
import tensorflow as tf


def convert_label2onehot(gt, num_classes):
    """
    Util function used to convert labels in the one-hot format
    :param gt: 4D tensor: [batch_size, image_width, image_height, 1]
    :return: 4D tensor: [batch_size, image_width, image_height, num_classes]
    """
    # Prep the data. Make sure the labels are in one-hot format
    gt_one_hot = tf.squeeze(gt, axis=3)
    gt_one_hot = tf.cast(gt_one_hot, tf.uint8)
    gt_one_hot = tf.one_hot(gt_one_hot, num_classes)
    return gt_one_hot


def convert_val2onehot(gt, num_classes):
    """
    Util function used to convert labels in the one-hot format
    :param gt: 3D tensor: [image_width, image_height, 1]
    :return: 3D tensor: [image_width, image_height, num_classes]
    """
    # Prep the data. Make sure the labels are in one-hot format
    gt_one_hot = tf.squeeze(gt, axis=2)
    gt_one_hot = tf.cast(gt_one_hot, tf.uint8)
    gt_one_hot = tf.one_hot(gt_one_hot, num_classes)
    return gt_one_hot


def compute_and_print_IoU_per_class(confusion_matrix, num_classes, is_incremental, from_new_class, to_new_class, class_mask=None):
    """
    Computes and prints mean intersection over union divided per class
    :param confusion_matrix: confusion matrix needed for the computation
    """
    logging.basicConfig(level=logging.INFO)
    mIoU = 0
    mean_class_acc_num = 0
    mean_pixel_acc_nobackground = 0
    mean_pixel_acc_new_classes = 0
    mIoU_nobackgroud = 0
    mIoU_new_classes = 0
    out = ''
    out_pixel_acc = ''
    index = ''
    true_classes = 0
    true_classes_pix = 0
    mean_class_acc_den = 0
    class_acc = 0
    mean_class_acc_num_nobgr = 0
    mean_class_acc_den_nobgr = 0
    mean_class_acc_sum_nobgr = 0
    mean_class_acc_sum=0
    if class_mask == None:
        class_mask = np.ones([num_classes], np.int8)
    for i in range(to_new_class+1):
        IoU = 0
        per_class_pixel_acc = 0
        class_acc = 0
        if class_mask[i] == 1:
            # IoU = true_positive / (true_positive + false_positive + false_negative)
            TP = confusion_matrix[i, i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i]) - TP
            # TN = np.sum(confusion_matrix) - TP - FP - FN

            denominator = (TP + FP + FN)
            # If the denominator is 0, we need to ignore the class.
            if denominator == 0 or i > to_new_class:
                denominator = 1
            else:
                true_classes += 1

            # per-class pixel accuracy
            per_class_pixel_acc = TP / (TP + FN)
            IoU = TP / denominator
            mIoU += IoU

            if i > 0:
                mIoU_nobackgroud += IoU

            if is_incremental:
                if (i >= from_new_class) and (i <= to_new_class):
                    mIoU_new_classes += IoU

            # mean class accuracy
            if not np.isnan(per_class_pixel_acc) and i <= to_new_class:
                mean_class_acc_num += TP
                mean_class_acc_den += TP + FN

                mean_class_acc_sum += per_class_pixel_acc
                true_classes_pix += 1

                if i > 0:
                    mean_class_acc_num_nobgr += TP
                    mean_class_acc_den_nobgr += TP + FN
                    mean_class_acc_sum_nobgr += per_class_pixel_acc

        index += '%7d' % i
        out += '%6.2f%%' % (IoU * 100)
        out_pixel_acc += '%6.2f%%' % (per_class_pixel_acc * 100)

    mIoU = mIoU / true_classes
    mean_pix_acc = mean_class_acc_num / mean_class_acc_den
    mean_pixel_acc_nobackground = mean_class_acc_num_nobgr / mean_class_acc_den_nobgr

    mIoU_nobackgroud = mIoU_nobackgroud / (true_classes - 1)

    if is_incremental:
        mIoU_new_classes = mIoU_new_classes / (to_new_class + 1 - from_new_class)

    logging.info(' index :     ' + index)
    logging.info(' class IoU : ' + out)
    logging.info(' class acc : ' + out_pixel_acc)
    logging.info(' mean pix acc : %.2f%%' % (mean_pix_acc * 100))
    logging.info(' mIoU : %.2f%%' % (mIoU * 100))
    logging.info(' mean_class_acc : %.2f%%' % ((mean_class_acc_sum/true_classes_pix) * 100))
    logging.info(' mIoU_nobackground : %.2f%%' % (mIoU_nobackgroud * 100))
    logging.info(' mean_pixel_acc_no_background : %.2f%%' % (mean_pixel_acc_nobackground * 100))
    if is_incremental:
        logging.info(' mIoU_new_classes : %.2f%%' % (mIoU_new_classes * 100))

    return mIoU_nobackgroud*100, mIoU_new_classes*100

def save_matlab_files(path, step, npy_data):
    """
    Save numpy data to a Matlab format
    :param step: step of the data
    :param npy_data: data to save
    """
    npy_data = npy_data[:, :, :, :]
    try:
        os.makedirs(path)
    except os.error:
        pass
    scipy.io.savemat(path + str(step) + '_softmax_output.mat', dict(x=npy_data))


def differentiable_argmax(logits):
    """
    Trick to obtain a differentiable argmax using softmax.

    :param logits: unprocessed tensor from the generator. 4D tensor: [batch_size, image_width, image_height, 3]
    :return: differentiable argmax of the imput logits. 4D tensor: [batch_size, image_width, image_height, 3]
    """
    with tf.variable_scope('differentiable_argmax'):
        y = tf.nn.softmax(logits)
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.one_hot(tf.argmax(y, 3), k), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
        return y
