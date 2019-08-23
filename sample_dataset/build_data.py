import os

import cv2
import tensorflow as tf

try:
    from os import scandir
except ImportError:
    # Python 2 polyfill module
    from scandir import scandir

# This file was build_data_supervised.py
FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('data_dir', '', 'data directory')
tf.flags.DEFINE_string('data_list', 'train_1_19.txt', 'data_list')
tf.flags.DEFINE_string('output_file', 'tfrecords/train_1_19.tfrecords',
                       'output tfrecords file')


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def read_labeled_image_list(data_dir, data_list):
    """
    Reads txt file containing paths to images and ground truth masks

    :param data_dir: path to the directory with images and masks
    :param data_list: path to the file with lines are the number of images
    :return: two lists with all file names for images and masks, respectively
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        try:
            image, mask = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image = mask = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    return images, masks


def data_writer(data_dir, data_list, output_file):
    """
    Write data to tfrecords
    """
    input_list, gt_list = read_labeled_image_list(data_dir, data_list)

    # create tfrecords dir if not exists
    output_dir = os.path.dirname(output_file)
    try:
        os.makedirs(output_dir)
    except os.error as e:
        pass

    images_num = len(input_list)

    # dump to tfrecords file
    writer = tf.python_io.TFRecordWriter(output_file)

    for i in range(images_num):

        # read image and ground truth
        input_path = input_list[i]
        input_path = input_path[1:]
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)

        height = image.shape[0]
        width = image.shape[1]
        with tf.gfile.FastGFile(input_path, 'rb') as f:
            image_data = f.read()
        gt_path = gt_list[i]
        gt_path = gt_path[1:]
        with tf.gfile.FastGFile(gt_path, 'rb') as f:
            gt_data = f.read()

        # Create a feature
        feature = {'input': _bytes_feature(image_data),
                   'gt': _bytes_feature(gt_data),
                   'height': _int64_feature(height),
                   'width': _int64_feature(width)}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

        if i % 100 == 0:
            print("Processed {}/{}.".format(i, images_num))
    print("Done.")
    writer.close()


def main(unused_argv):
    print("Convert data to tfrecords...")
    data_writer(FLAGS.data_dir, FLAGS.data_list, FLAGS.output_file)


if __name__ == '__main__':
    tf.app.run()
