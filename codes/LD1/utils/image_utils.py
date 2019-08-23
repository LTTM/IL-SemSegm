"""
Utility functions for image elaboration
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

'''
VOC2012 ground truth map:
0 = background -> [0, 0, 0]
1 = aeroplane -> [128, 0, 0]      
2 = bicycle -> [0, 128, 0]
3 = bird ->[128, 128, 0]
4 = boat ->[0, 0, 128]
5 = bottle ->[128, 0, 128]
6 = bus ->[0, 128, 128]
7 = car ->[128, 128, 128]
8 = cat ->[64, 0, 0]
9 = chair ->[192, 0, 0]
10 = cow ->[64, 128, 0]
11 = dining table ->[192, 128, 0]
12 = dog ->[64, 0, 128]
13 = horse ->[192, 0, 128]
14 = motorbike ->[64, 128, 128]
15 = person ->[192, 128, 128]
16 = potted plant ->[0, 64, 0]
17 = sheep ->[128, 64, 0]
18 = sofa ->[0, 192, 0]
19 = train ->[128, 192, 0]
20 = tv/monitor ->[0, 64, 128]
'''

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)


def convert_output2rgb(image):
    """
    Convert the output tensor of the generator to an RGB image

    :param image: output tensor given by the generator. 4D tensor: [batch_size, image_width, image_height, num_classes]
    :return: image converted to RGB format. 4D tensor: [batch_size, image_width, image_height, 3]
    """
    table = tf.constant([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
                         [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
                         [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
                         [128, 192, 0], [0, 64, 128]], tf.uint8)

    labels = tf.argmax(image, axis=3)
    out_RGB = tf.nn.embedding_lookup(table, labels)
    return out_RGB


def convert_gt2rgb(image):
    """
    Convert the tensor containing the GT to an RGB image

    :param image: tensor containing gt labels. 4D tensor: [batch_size, image_width, image_height, 1]
    :return: image converted to RGB format. 4D tensor: [batch_size, image_width, image_height, 3]
    """
    table = tf.constant([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
                         [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
                         [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
                         [128, 192, 0], [0, 64, 128],
                         # fix wrong classes using the value of the background (Only for better visualization of gt)
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                         [0, 0, 0], [0, 0, 0], [255, 255, 255]], tf.uint8)

    labels = tf.squeeze(image, axis=3)
    labels = tf.cast(labels, tf.int32)
    out_RGB = tf.nn.embedding_lookup(table, labels)
    return out_RGB


def preprocess_train_data(image, labels, image_size, seed=12345):
    """
    Pre process training data to fit the net requirements

    :param image: image to process. 3D tensor: [image_width, image_height, 3]
    :param labels: labels corresponding to image. 3D tensor: [image_width, image_height, 1]
    :param image_size: size of the image
    :return: modified image and label
    """
    image = tf.cast(image, tf.float32)

    # Extract mean.
    image = convert2float(image)

    # Randomly scale the images and labels.
    image, labels = random_image_scaling(image, labels, seed)

    # Randomly mirror the images and labels.
    image, labels = image_mirroring(image, labels, seed)

    # Randomly crop image to the right size
    image, labels = random_crop_and_pad_image_and_labels(image, labels, image_size, image_size, seed)

    return image, labels


def preprocess_val_data(image, labels):
    """
    Pre process validation data to fit the net requirements

    :param image: image to process. 3D tensor: [image_width, image_height, 3]
    :param labels: labels corresponding to image. 3D tensor: [image_width, image_height, 1]
    :return: modified image and label
    """
    image = tf.cast(image, tf.float32)

    image = convert2float(image)

    labels = tf.cast(labels, tf.float32)

    return image, labels


def convert2float(image):
    """
    Transform the input image: convert to float, subtract the mean computed on the training data

    :param image: image to convert. 3D tensor: [image_width, image_height, 3]
    :return: modified image
    """
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image - IMG_MEAN


def random_image_scaling(img, label, seed):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    :param img: image to scale. 3D tensor: [image_width, image_height, 3]
    :param label: label corresponding to image. 3D tensor: [image_width, image_height, 1]
    :return: modified image and label
    """
    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=seed)

    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))

    new_shape = tf.squeeze(tf.stack([h_new, w_new]), axis=1)
    img = tf.image.resize_images(img, new_shape, method=ResizeMethod.BILINEAR)
    label = tf.image.resize_images(label, new_shape, method=ResizeMethod.NEAREST_NEIGHBOR)

    return img, label


def image_mirroring(img, label, seed):
    """
    Randomly mirror the image

    :param img: image to mirror. 3D tensor: [image_width, image_height, 3]
    :param label: label corresponding to image. 3D tensor: [image_width, image_height, 1]
    :return: modified image and label
    """

    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32, seed=seed)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label


def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, seed, ignore_label=255):
    """
    Randomly scale and crop and pads the input images.

    :param image: training image to crop/ pad. 3D tensor: [image_width, image_height, 3]
    :param label: segmentation mask to crop/ pad. 3D tensor: [image_width, image_height, 1]
    :param crop_h: height of cropped segment
    :param crop_w: width of cropped segment
    :param ignore_label: label to ignore during the training
    :return: modified image and label
    """

    # Pad if needed
    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label  # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]),
                                                tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 4], seed=seed)
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h, crop_w, 1))
    return img_crop, label_crop
