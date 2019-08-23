#######################################################################################
# Michieli U., Zanuttigh P.                                                           #
# "Incremental Learning Techniques for Semantic Segmentation"                         #
# Proceedings of the International Conference on Computer Vision (ICCV)               #
# Workshop on Transferring and Adapting Source Knowledge in Computer Vision (TASK-CV) #
# Seoul (South Korea), 2019.                                                          #
#######################################################################################

import logging

import numpy as np
import tensorflow as tf

from model import model
from utils import tensor_utils

FLAGS = tf.flags.FLAGS

# Utilized data
tf.flags.DEFINE_string('training_file', '../../dataset/VOC2012_tfrecords/train.tfrecords',
                       'tfrecords file for training')
tf.flags.DEFINE_string('validation_file', '../../dataset/VOC2012_tfrecords/val.tfrecords', 'tfrecords file for validation')
tf.flags.DEFINE_string('pretrained_folder', 'pretrained', 'path of the folder for pretraied encoder models')

# Data format
tf.flags.DEFINE_integer('batch_size', 4, 'batch size')
tf.flags.DEFINE_integer('validation_batch_size', 4, ' validation batch size default: 1449')
tf.flags.DEFINE_integer('image_size', 321, 'image size, default: 256')
tf.flags.DEFINE_integer('num_classes', 21, 'number of output classes')
tf.flags.DEFINE_string('norm', 'batch', '[instance, batch] use instance norm or batch norm, default: instance')

# Training parameters
tf.flags.DEFINE_integer('training_steps', 20000, 'number of training steps')
tf.flags.DEFINE_integer('logging_steps_interval', 1, 'interval of steps between each log')
tf.flags.DEFINE_integer('validation_steps_interval', 1, 'interval of steps between each validation')
tf.flags.DEFINE_integer('save_steps_interval', 10000, 'interval of steps between each log')

# Checkpoint folder
tf.flags.DEFINE_string('load_model',
                       'checkpoints/20190205-0844',
                       'folder of saved model that you wish to continue training (checkpoint/{})')

# Network optimization parameter (segmentation network uses SGD optimizer)
tf.flags.DEFINE_float('start_learning_rate', 2.5e-4, 'initial learning rate for segmentation network')
tf.flags.DEFINE_float('end_learning_rate', 0.0, 'final decay rate for segmentation network optimizer')
tf.flags.DEFINE_integer('start_decay_step', 0, 'number of steps after which start decay for segmentation network')
tf.flags.DEFINE_integer('decay_steps', 20000, 'number of steps of decay learning rate for segmentation network')
tf.flags.DEFINE_float('decay_power', 0.9, 'power of weight decay for segmentation network optimizer')
tf.flags.DEFINE_float('momentum', 0.9, 'momentum for segmentation network optimizer')
tf.flags.DEFINE_float('weight_decay', 1e-4, 'weight decay for momentum of segmentation network optimizer')



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
            end_learning_rate=FLAGS.end_learning_rate
        )

        # Define the model
        G_sup_loss, G_sup_optimizer, G_output_maps, training_summary_sup = gan.training_model()
        iterator, mean_validation_loss, mean_validation_update_op, confusion_matrix, accuracy, accuracy_update_op, mean_IoU, mean_IoU_update_op, validation_summary = gan.validation_model()
		
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
            # Compute validation metrics
            sess.run(tf.local_variables_initializer())
            sess.run([iterator.initializer])
            n = 0
            local_confusion_matrix = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.int)
            for i in range(FLAGS.validation_batch_size):
                try:
                    cm, _, _, _ = sess.run(
                        [confusion_matrix, mean_validation_update_op, mean_IoU_update_op, accuracy_update_op])
                    local_confusion_matrix += cm
                    n += 1
                except tf.errors.OutOfRangeError:  # no more items
                    logging.info(
                        ' The chosen value ({}) exceed number of validation images! Used {} instead.'.format(
                            FLAGS.validation_batch_size, n))
                    break
            mean_validation_loss_val, mean_IoU_val, accuracy_val, summary = sess.run(
                [mean_validation_loss, mean_IoU, accuracy, validation_summary])
            logging.info(' Validation_loss : {}'.format(mean_validation_loss_val))
            logging.info(' accuracy : {:0.2f}%'.format(accuracy_val * 100))
            tensor_utils.compute_and_print_IoU_per_class(local_confusion_matrix, FLAGS.num_classes)
            train_writer.add_summary(summary, step)
            train_writer.flush()

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
