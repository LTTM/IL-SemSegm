#######################################################################################
# Michieli U., Zanuttigh P.                                                           #
# "Incremental Learning Techniques for Semantic Segmentation"                         #
# Proceedings of the International Conference on Computer Vision (ICCV)               #
# Workshop on Transferring and Adapting Source Knowledge in Computer Vision (TASK-CV) #
# Seoul (South Korea), 2019.                                                          #
#######################################################################################

import logging
import os
import numpy as np
import tensorflow as tf

from model import model
from utils import tensor_utils

FLAGS = tf.flags.FLAGS

# Utilized data
tf.flags.DEFINE_string('training_file', '../.../../dataset/VOC2012_tfrecords/train_20.tfrecords',
                       'tfrecords file for training')
tf.flags.DEFINE_string('validation_file', '../../../dataset/VOC2012_tfrecords/train_20.tfrecords', 'tfrecords file for validation')
tf.flags.DEFINE_string('pretrained_folder', 'pretrained', 'path of the folder for pretraied encoder models')

# Data format
tf.flags.DEFINE_integer('batch_size', 1, 'batch size')
tf.flags.DEFINE_integer('validation_batch_size', 50000, ' validation batch size default: 1449')
tf.flags.DEFINE_integer('image_size', 321, 'image size, default: 256')
tf.flags.DEFINE_integer('num_classes', 21, 'number of output classes')
tf.flags.DEFINE_string('norm', 'batch', '[instance, batch] use instance norm or batch norm, default: instance')

# Training parameters
tf.flags.DEFINE_integer('training_steps', 0, 'number of training steps')
tf.flags.DEFINE_integer('logging_steps_interval', 1, 'interval of steps between each log')
tf.flags.DEFINE_integer('validation_steps_interval', 1, 'interval of steps between each validation')
tf.flags.DEFINE_integer('save_steps_interval', 0, 'interval of steps between each log')

# Checkpoint folder
tf.flags.DEFINE_string('load_model',
                       '1_to_19',
                       'folder of saved model that you wish to continue training (checkpoint/{})')

# Network optimization parameter (segmentation network uses SGD optimizer)
tf.flags.DEFINE_float('start_learning_rate', 0, 'initial learning rate for segmentation network')
tf.flags.DEFINE_float('end_learning_rate', 0.0, 'final decay rate for segmentation network optimizer')
tf.flags.DEFINE_integer('start_decay_step', 0, 'number of steps after which start decay for segmentation network')
tf.flags.DEFINE_integer('decay_steps', 0, 'number of steps of decay learning rate for segmentation network')
tf.flags.DEFINE_float('decay_power', 0, 'power of weight decay for segmentation network optimizer')
tf.flags.DEFINE_float('momentum', 0, 'momentum for segmentation network optimizer')
tf.flags.DEFINE_float('weight_decay', 0, 'weight decay for momentum of segmentation network optimizer')


# Inremental learning parameters
tf.flags.DEFINE_float('lambda_distillation', None, 'weight of the distillation loss. Expected: 0<=lambda_distillation')
tf.flags.DEFINE_integer('is_incremental', None, 'Determine whether we are training in incremental way [1] or not [0]')
tf.flags.DEFINE_integer('is_save_new_softmax_maps', 0, 'Determine whether to save new softmax maps [1] or not [0]')


def test():

    # Prevent error
    assert FLAGS.load_model is not None, 'Needed pre-trained model for test!'

    checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(12345)
        gan = model(
            train_file=FLAGS.training_file,
            validation_file=FLAGS.validation_file,
            batch_size=FLAGS.batch_size,
            image_size=FLAGS.image_size,
            num_classes=FLAGS.num_classes,
            pretrained_folder=FLAGS.pretrained_folder,
            start_learning_rate=FLAGS.start_learning_rate,
            momentum=FLAGS.momentum,
            weight_decay=FLAGS.weight_decay,
            decay_power=FLAGS.decay_power,
            decay_steps=FLAGS.decay_steps,
            start_decay_step=FLAGS.start_decay_step,
            end_learning_rate=FLAGS.end_learning_rate,
        )

        # Define the model
        iterator, mean_validation_loss, mean_validation_update_op, confusion_matrix, accuracy, accuracy_update_op,\
        mean_IoU, mean_IoU_update_op, G_output_maps, G_features_maps, next_images, next_labels, validation_summary = gan.validation_model()
		
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()

    with tf.Session(graph=graph, config=None) as sess:
        tf.set_random_seed(12345)

        # Load model
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
        meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
        loader = tf.train.import_meta_graph(meta_graph_path)
        loader.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
        start_step = int(meta_graph_path.split("-")[-1].split(".")[0])
        step = start_step

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            os.makedirs('softmax_output/')
            os.makedirs('next_images/')
            os.makedirs('next_labels/')
            os.makedirs('feature_maps/')
        except os.error:
            pass

        try:
            # Compute validation metrics
            sess.run(tf.local_variables_initializer())
            sess.run([iterator.initializer])
            n = 0
            for i in range(FLAGS.validation_batch_size):
                try:
                    print(str(i))
                    output_maps, features_maps, next_images_run, next_labels_run = sess.run([G_output_maps, G_features_maps, next_images, next_labels])
                    output_maps = np.array(output_maps)
                    next_images_run = np.expand_dims(np.array(next_images_run), axis=0)
                    next_labels_run = np.expand_dims(np.array(next_labels_run), axis=0)

                    features_maps = np.array(features_maps)
					
                    np.save('softmax_output/'+str('{:04d}'.format(i))+'_softmax_output.npy', output_maps)
                    np.save('feature_maps/' + str('{:04d}'.format(i)) + '_feature_maps.npy', features_maps)
                    np.save('next_images/' + str('{:04d}'.format(i)) + '_images.npy', next_images_run)
                    np.save('next_labels/' + str('{:04d}'.format(i)) + '_labels.npy', next_labels_run)

                    n += 1
                except tf.errors.OutOfRangeError:  # no more items
                    logging.info(
                        ' The chosen value ({}) exceed number of train20 images! Used {} instead.'.format(
                            FLAGS.validation_batch_size, n))
                    break

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)

def main(unused_argv):
    test()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
