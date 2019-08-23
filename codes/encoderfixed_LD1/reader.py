#######################################################################################
# Michieli U., Zanuttigh P.                                                           #
# "Incremental Learning Techniques for Semantic Segmentation"                         #
# Proceedings of the International Conference on Computer Vision (ICCV)               #
# Workshop on Transferring and Adapting Source Knowledge in Computer Vision (TASK-CV) #
# Seoul (South Korea), 2019.                                                          #
#######################################################################################

import matplotlib.pyplot as plt
import tensorflow as tf

from utils import image_utils

# Was taken from ReaderSupervised
class Reader:
    def __init__(self, tfrecords_file, image_size,
                 min_queue_examples, batch_size, name, num_threads=8, seed=12345):
        """
        Args:
          tfrecords_file: string, tfrecords file path
          min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
          batch_size: integer, number of images per batch
          num_threads: integer, number of preprocess threads
        """
        self.tfrecords_file = tfrecords_file
        self.image_size = image_size
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.seed = seed
        self.name = name
        """
        Returns:
          
        """

    def feed(self):
        """
        Read and return the information contained in the tfrecords file

        :return:
                input_images: 4D tensor [batch_size, image_width, image_height, image_depth],
                gt_images: 4D tensor [batch_size, image_width, image_height, image_depth],
                summary: tensorboard summary
        """
        with tf.name_scope(self.name):
            tf.set_random_seed(self.seed)
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])
            reader = tf.TFRecordReader()

            _, serialized_example = reader.read(filename_queue)
            sample = tf.parse_single_example(
                serialized_example,
                features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'input': tf.FixedLenFeature([], tf.string),
                    'gt': tf.FixedLenFeature([], tf.string)
                })

            # Obtain back image and gt
            height = tf.cast(sample['height'], tf.int32)
            width = tf.cast(sample['width'], tf.int32)
            image = tf.image.decode_jpeg(sample['input'], channels=3)
            gt = tf.image.decode_png(sample['gt'], channels=1)

            # Preprocess paper_images
            image, gt = image_utils.preprocess_train_data(image, gt, self.image_size, self.seed)

            # Construct output by shuffling batch
            input_images, gt_images = tf.train.shuffle_batch(
                [image, gt], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=self.min_queue_examples,
                seed=self.seed
            )

        # Summary
        images_slice = tf.slice(input_images, [0, 0, 0, 0], [1, self.image_size, self.image_size, 3])
        gt_slice = image_utils.convert_gt2rgb(
            tf.slice(gt_images, [0, 0, 0, 0], [1, self.image_size, self.image_size, 1]))
        summary = tf.summary.image(self.name + '/input', images_slice)
        return input_images, gt_images, summary


'''def test_reader():
    TRAIN_FILE = '../../data/tfrecords/VOC2012/train.tfrecords'

    with tf.Graph().as_default():
        reader = Reader(TRAIN_FILE, 321, 1000, 2, 'test')
        input_images, gt_images, _ = reader.feed()

        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            step = 0
            while not coord.should_stop() and step < 10:
                img1, img2 = sess.run([input_images, gt_images])
                step += 1
                plt.subplot(1, 2, 1)
                plt.imshow(img1[0])
                plt.subplot(1, 2, 2)
                img = image_utils.convert_gt2rgb(img2)
                img = tf.slice(img, [0, 0, 0, 0], [1, 321, 321, 3])
                img = tf.reshape(img, [1, 321, 321, 3])
                img = sess.run([img])
                plt.imshow(img[0][0])
                plt.show()
        except KeyboardInterrupt:
            print('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    test_reader()'''
