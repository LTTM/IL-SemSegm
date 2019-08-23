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
from matplotlib import pyplot as plt
from model_with_distillation import model
from utils import tensor_utils

FLAGS = tf.flags.FLAGS

# Utilized data
tf.flags.DEFINE_string('training_file', '../../../dataset/VOC2012_tfrecords/val_1_20.tfrecords',
                       'tfrecords file for training')
tf.flags.DEFINE_string('validation_file', '../../../dataset/VOC2012_tfrecords/val_1_20.tfrecords', 'tfrecords file for validation')
tf.flags.DEFINE_integer('up_to_class', None, 'up to number of classes to train on (background always included)')
tf.flags.DEFINE_string('pretrained_folder', '../../../pretrained', 'path of the folder for pretrained encoder models')

# Data format
tf.flags.DEFINE_integer('batch_size', 4, 'batch size')
tf.flags.DEFINE_integer('validation_batch_size', 99999999, ' validation batch size default: 1449')
tf.flags.DEFINE_integer('image_size', 321, 'image size, default: 256')
tf.flags.DEFINE_integer('num_classes', 21, 'number of output classes')
tf.flags.DEFINE_string('norm', 'batch', '[instance, batch] use instance norm or batch norm, default: instance')

# Training parameters
tf.flags.DEFINE_integer('training_steps', 40000, 'number of training steps')
tf.flags.DEFINE_integer('logging_steps_interval', 50, 'interval of steps between each log')
tf.flags.DEFINE_integer('validation_steps_interval', 1000, 'interval of steps between each validation')
tf.flags.DEFINE_integer('save_steps_interval', 1000, 'interval of steps between each log')

# Checkpoint folder
tf.flags.DEFINE_string('load_model', './checkpoints/1_to_10_11_11_distill_lambda1.0_std_new_old_dist_all',
                       'folder of saved model that you wish to continue training (checkpoints/{})')

# Network optimization parameter (segmentation network uses SGD optimizer)
tf.flags.DEFINE_float('start_learning_rate', 2.5e-4, 'initial learning rate for segmentation network')
tf.flags.DEFINE_float('end_learning_rate', 0.0, 'final decay rate for segmentation network optimizer')
tf.flags.DEFINE_integer('start_decay_step', 0, 'number of steps after which start decay for segmentation network')
tf.flags.DEFINE_integer('decay_steps', 20000, 'number of steps of decay learning rate for segmentation network')
tf.flags.DEFINE_float('decay_power', 0.9, 'power of weight decay for segmentation network optimizer')
tf.flags.DEFINE_float('momentum', 0.9, 'momentum for segmentation network optimizer')
tf.flags.DEFINE_float('weight_decay', 1e-4, 'weight decay for momentum of segmentation network optimizer')

# Incremental learning parameters
tf.flags.DEFINE_float('lambda_distillation', None, 'weight of the distillation loss. Expected: 0<=lambda_distillation')
tf.flags.DEFINE_integer('is_incremental', 1, 'determine whether we are training in incremental way [1] or not [0]')
tf.flags.DEFINE_integer('is_save_new_softmax_maps', 0, 'determine whether to save new softmax maps [1] or not [0]')
tf.flags.DEFINE_integer('from_new_class', 11, 'specifies the starting index [included] of the new classes to the added')
tf.flags.DEFINE_integer('to_new_class', 11, 'specifies the ending index [included] of the new classes to be added')
tf.flags.DEFINE_string('standard_loss_applied_to', 'all', 'specifies where to apply the standard loss: [all, new, new_old] classes')
tf.flags.DEFINE_string('distill_loss_applied_to', 'all', 'specifies where to apply the distillation loss: [all, new, old] classes')


def test():

    # Prevent error
    assert FLAGS.load_model is not None, 'Needed pre-trained model for test!'

    checkpoints_dir = FLAGS.load_model
    graph = tf.Graph()
    # Define the model
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
            lambda_distillation=FLAGS.lambda_distillation,
            is_incremental=FLAGS.is_incremental,
            from_new_class=FLAGS.from_new_class,
            to_new_class=FLAGS.to_new_class,
            standard_loss_applied_to=FLAGS.standard_loss_applied_to,
            distill_loss_applied_to=FLAGS.distill_loss_applied_to,
        )

        pred_img, gt, iterator, mean_validation_loss, mean_validation_update_op, confusion_matrix, accuracy, accuracy_update_op, \
        mean_IoU, mean_IoU_update_op, G_output_maps, validation_summary = gan.validation_model()

        percentage_pixels_not_bgr_ = tf.placeholder(tf.float32, shape=())
        mIoU_no_bgr_ = tf.placeholder(tf.float32, shape=())
        mIoU_new_classes_ = tf.placeholder(tf.float32, shape=())
        percentage_pixels_not_bgr_summary = tf.summary.scalar('validation/percentage_pixels_not_bgr',
                                                              percentage_pixels_not_bgr_)
        summary_mIoU_no_bgr = tf.summary.scalar('validation/mIoU_not_bgr', mIoU_no_bgr_)
        summary_mIoU_new_classes = tf.summary.scalar('validation/mIoU_new_classes', mIoU_new_classes_)

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
            if step % FLAGS.validation_steps_interval == 0:
                sess.run(tf.local_variables_initializer())
                sess.run([iterator.initializer])
                n = 0
                local_confusion_matrix = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.int)
                non_zero_pixels_pred = non_zero_pixels_gt = total_pixels = 0

                for i in range(FLAGS.validation_batch_size):
                    try:
                        cm, _, _, _, pred_img_value, gt_value = sess.run([confusion_matrix,
                        mean_validation_update_op, mean_IoU_update_op, accuracy_update_op, pred_img, gt])
                        local_confusion_matrix += cm

                        non_zero_pixels_pred += np.count_nonzero(pred_img_value)
                        non_zero_pixels_gt += np.count_nonzero(gt_value)
                        total_pixels += pred_img_value.size
                        n += 1
                    except tf.errors.OutOfRangeError:  # no more items
                        logging.info(
                            ' The chosen value ({}) exceed number of validation images! Used {} instead.'.format(
                                FLAGS.validation_batch_size, n))
                        break
                mean_validation_loss_value, mean_validation_IoU_value, \
                mean_validation_accuracy_value, summary = sess.run([mean_validation_loss, mean_IoU,
                                                                    accuracy, validation_summary])
                train_writer.add_summary(summary, step)

                percentage_pixels_not_bgr = (non_zero_pixels_pred / total_pixels) * 100
                percentage_pixels_not_bgr_gt = (non_zero_pixels_gt / total_pixels) * 100  # It is constant
                train_writer.add_summary(sess.run(percentage_pixels_not_bgr_summary,
                                                  feed_dict={percentage_pixels_not_bgr_: percentage_pixels_not_bgr}),
                                         step)

                mIoU_no_bgr, mIoU_new_classes = tensor_utils.compute_and_print_IoU_per_class(local_confusion_matrix,
                                                                                             FLAGS.num_classes,
                                                                                             FLAGS.is_incremental,
                                                                                             FLAGS.from_new_class,
                                                                                             FLAGS.to_new_class)

                train_writer.add_summary(sess.run(summary_mIoU_no_bgr, feed_dict={mIoU_no_bgr_: mIoU_no_bgr}), step)
                train_writer.add_summary(
                    sess.run(summary_mIoU_new_classes, feed_dict={mIoU_new_classes_: mIoU_new_classes}), step)

                logging.info(' Validation_loss : {}'.format(mean_validation_loss_value))
                logging.info(' accuracy TF: {:0.2f}%'.format(mean_validation_accuracy_value * 100))
                logging.info(' mIoU TF : {:0.2f}%'.format(mean_validation_IoU_value * 100))
                logging.info(' Percentage of pixels not background : {:0.2f}%'.format(percentage_pixels_not_bgr))
                logging.info(
                    ' Percentage of pixels not background (GT) : {:0.2f}%'.format(percentage_pixels_not_bgr_gt))
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
