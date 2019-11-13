#######################################################################################
# Michieli U., Zanuttigh P.                                                           #
# "Incremental Learning Techniques for Semantic Segmentation"                         #
# Proceedings of the International Conference on Computer Vision (ICCV)               #
# Workshop on Transferring and Adapting Source Knowledge in Computer Vision (TASK-CV) #
# Seoul (South Korea), 2019.                                                          #
#######################################################################################

import logging
import math
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
from model_with_distillation import model
from utils import tensor_utils

FLAGS = tf.flags.FLAGS

# Utilized data
tf.flags.DEFINE_string('training_file', '../../dataset/VOC2012_tfrecords/train_20.tfrecords',
                       'tfrecords file for training')
tf.flags.DEFINE_string('validation_file', '../../dataset/VOC2012_tfrecords/val_1_20.tfrecords', 'tfrecords file for validation')
tf.flags.DEFINE_integer('up_to_class', None, 'up to number of classes to train on (background always included)')
tf.flags.DEFINE_string('pretrained_folder', '../../pretrained', 'path of the folder for pretrained encoder models')

# Data format
tf.flags.DEFINE_integer('batch_size', 4, 'batch size')
tf.flags.DEFINE_integer('validation_batch_size', 1449, ' validation batch size default: 1449')
tf.flags.DEFINE_integer('image_size', 321, 'image size, default: 256')
tf.flags.DEFINE_integer('num_classes', 21, 'number of output classes')
tf.flags.DEFINE_string('norm', 'batch', '[instance, batch] use instance norm or batch norm, default: instance')

# Training parameters
tf.flags.DEFINE_integer('training_steps', 40000, 'number of training steps')
tf.flags.DEFINE_integer('logging_steps_interval', 50, 'interval of steps between each log')
tf.flags.DEFINE_integer('validation_steps_interval', 1000, 'interval of steps between each validation')
tf.flags.DEFINE_integer('save_steps_interval', 1000, 'interval of steps between each log')

# Checkpoint folder
tf.flags.DEFINE_string('load_model', None,
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
tf.flags.DEFINE_integer('is_incremental', None, 'determine whether we are training in incremental way [1] or not [0]')
tf.flags.DEFINE_integer('is_save_new_softmax_maps', 0, 'determine whether to save new softmax maps [1] or not [0]')
tf.flags.DEFINE_integer('from_new_class', None, 'specifies the starting index [included] of the new classes to the added')
tf.flags.DEFINE_integer('to_new_class', None, 'specifies the ending index [included] of the new classes to be added')
tf.flags.DEFINE_string('standard_loss_applied_to', 'all', 'specifies where to apply the standard loss: [all, new, new_old] classes')
tf.flags.DEFINE_string('distill_loss_applied_to', 'old_bgr', 'specifies where to apply the distillation loss: [all, new, old] classes')
# NB: the default of the above two parameters must remain 'all', see network_loss the reason why
tf.flags.DEFINE_integer('is_distillonfeatures', 0, 'determine whether we are distilling on the features [1] or on the softmax output [0]')

def train():
    assert (FLAGS.is_incremental is not None), "is_incremental must always be specified!"
    assert (FLAGS.lambda_distillation is not None), "lambda_distillation must always be specified! (for the sake of clarity)"

    # Define checkpoint folder name
    if not FLAGS.is_incremental:
        assert (FLAGS.load_model is None), "If is_incremental=false, cannot be specified load_model!"
        assert (FLAGS.up_to_class is not None), "If is_incremental=false, up_to_class must be specified!"
        checkpoints_dir = checkpoints_dir_save = "checkpoints/1_to_" + str(FLAGS.up_to_class)
        FLAGS.to_new_class = FLAGS.up_to_class
    else: # is_incremental is True
        assert (FLAGS.load_model is not None), "If is_incremental=true, must be specified load_model!"
        assert (FLAGS.from_new_class is not None), "If is_incremental=true, must be specified from_new_class!"
        assert (FLAGS.to_new_class is not None), "If is_incremental=true, must be specified to_new_class!"
        if FLAGS.lambda_distillation != 0:
            checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
            checkpoints_dir_save = checkpoints_dir + "_{}_{}_distill_lambda{}_std_{}_dist_{}".format(FLAGS.from_new_class,
                FLAGS.to_new_class, FLAGS.lambda_distillation, FLAGS.standard_loss_applied_to, FLAGS.distill_loss_applied_to)
        else:
            checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
            checkpoints_dir_save = checkpoints_dir + "_{}_{}_nodistill_std_{}".format(
                FLAGS.from_new_class, FLAGS.to_new_class, FLAGS.standard_loss_applied_to)

    try:
        os.makedirs(checkpoints_dir)
        os.makedirs(checkpoints_dir_save)
    except os.error:
        pass

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
            is_distillonfeatures=FLAGS.is_distillonfeatures,
            distill_loss_applied_to=FLAGS.distill_loss_applied_to,
        )

        G_sup_loss, G_distill_loss, G_final_loss, G_sup_optimizer, training_summary_sup, G_new_softmax, G_new_features = gan.training_model()

        pred_img, gt, iterator, mean_validation_loss, mean_validation_update_op, confusion_matrix, accuracy, accuracy_update_op, \
        mean_IoU, mean_IoU_update_op, G_output_maps, validation_summary = gan.validation_model()

        percentage_pixels_not_bgr_ = tf.placeholder(tf.float32, shape=())
        mIoU_no_bgr_ = tf.placeholder(tf.float32, shape=())
        mIoU_new_classes_ = tf.placeholder(tf.float32, shape=())
        percentage_pixels_not_bgr_summary = tf.summary.scalar('validation/percentage_pixels_not_bgr',
                                                              percentage_pixels_not_bgr_)
        summary_mIoU_no_bgr = tf.summary.scalar('validation/mIoU_not_bgr', mIoU_no_bgr_)
        summary_mIoU_new_classes = tf.summary.scalar('validation/mIoU_new_classes', mIoU_new_classes_)

        train_writer = tf.summary.FileWriter(checkpoints_dir_save, graph)
        saver = tf.train.Saver()
    summary = None
    G_loss_value = 0
    step = 0
    start_step = 0

    # Load checkpoint if provided
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        tf.set_random_seed(12345)
        if FLAGS.load_model is not None:  # Restore pre trained checkpoint
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
            loader = tf.train.import_meta_graph(meta_graph_path)
            loader.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            start_step = int(meta_graph_path.split("-")[-1].split(".")[0])
            step = start_step
        else:  # Load pre trained encoder
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            loader = tf.train.Saver(var_list=gan.G.encoder_var, reshape=True)
            loader.restore(sess, FLAGS.pretrained_folder + '/deeplab_resnet_init.ckpt')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        new_soft_mat_dir=None
        if FLAGS.is_incremental:
            num_new_images = 0
            count = 0
            if FLAGS.is_save_new_softmax_maps:
                try:
                    new_soft_mat_dir = 'new_softmax_mat_'+str(FLAGS.from_new_class)+'_'+str(FLAGS.to_new_class)
                    new_features_mat_dir = 'new_features_mat_'+str(FLAGS.from_new_class)+'_'+str(FLAGS.to_new_class)
                    os.makedirs(new_soft_mat_dir)
                    os.makedirs(new_features_mat_dir)
                except os.error:
                    pass

        # Train
        try:
            while not coord.should_stop() and step < FLAGS.training_steps:

                if FLAGS.is_incremental:
                    new_batch_images = np.zeros([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 3])
                    new_batch_labels = np.zeros([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1])
                    new_batch_output_maps = np.zeros([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, FLAGS.num_classes])
                    new_batch_feature_maps = np.zeros([FLAGS.batch_size, 41, 41, 2048])
                    for i in range(FLAGS.batch_size):
                        new_batch_images[i, :, :, :] = np.load('next_images/' + '{:04d}'.format(FLAGS.batch_size * num_new_images + i) + '_images.npy')
                        new_batch_labels[i, :, :, :] = np.load('next_labels/' + '{:04d}'.format(FLAGS.batch_size * num_new_images + i) + '_labels.npy')
                        new_batch_output_maps[i, :, :, :] = np.expand_dims(np.load('softmax_output/' + '{:04d}'.format(FLAGS.batch_size * num_new_images + i)
                                                                                   + '_softmax_output.npy'), axis=0)
                        new_batch_feature_maps[i, :, :, :] = np.load('feature_maps/' + '{:04d}'.format(FLAGS.batch_size * num_new_images + i)
                                                                     + '_feature_maps.npy')

                    dividend = math.floor(len(os.listdir('next_labels'))/FLAGS.batch_size)
                    num_new_images = (num_new_images + 1) % dividend

                    # load old batch, and feed dict together with other data
                    _, G_loss_value, summary, new_softmax, new_features, G_distill_loss_value, G_final_loss_value  =(sess.run(
                        [G_sup_optimizer, G_sup_loss, training_summary_sup, G_new_softmax, G_new_features, G_distill_loss, G_final_loss],
                        feed_dict={gan.network_input: new_batch_images,
                                   gan.network_input_labels: new_batch_labels,
                                   gan.network_input_output_maps: new_batch_output_maps,
                                   gan.network_input_feature_maps: new_batch_feature_maps}))

                    if count <= dividend and FLAGS.is_save_new_softmax_maps:
                        # print(count)
                        new_softmax = np.array(new_softmax)
                        new_features = np.array(new_features)
                        tensor_utils.save_matlab_files(new_soft_mat_dir+'/', '{:04d}'.format(count*FLAGS.batch_size), new_softmax)
                        tensor_utils.save_matlab_files(new_features_mat_dir+'/', '{:04d}'.format(count*FLAGS.batch_size), new_features)
                        count += 1

                    train_writer.add_summary(summary, step)
                    train_writer.flush()

                    # Output computed metrics
                    if step % FLAGS.logging_steps_interval == 0:
                        logging.info('-----------Step %d:-------------' % step)
                        logging.info(' Standard_loss : {}'.format(G_loss_value))
                        if isinstance(G_distill_loss_value, list):
                            logging.info(' Distillation_loss : 0 [empty list returned]')
                        else:
                            logging.info(' Distillation_loss : {}'.format(G_distill_loss_value))
                        logging.info(' Final_loss : {}'.format(G_final_loss_value))

                # If not incremental_training, train using the train_tfrecord specified
                else:
                    _, G_loss_value, G_final_loss_value, summary = (sess.run([G_sup_optimizer, G_sup_loss, G_final_loss, training_summary_sup]))
                    train_writer.add_summary(summary, step)
                    train_writer.flush()
                    # Output computed metrics
                    if step % FLAGS.logging_steps_interval == 0:
                        logging.info('-----------Step %d:-------------' % step)
                        logging.info(' Standard_loss=Final_loss : {}'.format(G_final_loss_value))

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
                    mean_validation_loss_value, mean_validation_IoU_value,\
                    mean_validation_accuracy_value, summary = sess.run([mean_validation_loss, mean_IoU,
                                                                        accuracy, validation_summary])
                    train_writer.add_summary(summary, step)

                    percentage_pixels_not_bgr = (non_zero_pixels_pred/total_pixels) * 100
                    percentage_pixels_not_bgr_gt = (non_zero_pixels_gt / total_pixels) * 100 # It is constant
                    train_writer.add_summary(sess.run(percentage_pixels_not_bgr_summary,
                                                      feed_dict={percentage_pixels_not_bgr_: percentage_pixels_not_bgr}),step)

                    mIoU_no_bgr, mIoU_new_classes = tensor_utils.compute_and_print_IoU_per_class(local_confusion_matrix,
                                 FLAGS.num_classes, FLAGS.is_incremental, FLAGS.from_new_class, FLAGS.to_new_class)
                    train_writer.add_summary(sess.run(summary_mIoU_no_bgr, feed_dict={mIoU_no_bgr_: mIoU_no_bgr}), step)
                    train_writer.add_summary(sess.run(summary_mIoU_new_classes, feed_dict={mIoU_new_classes_: mIoU_new_classes}), step)

                    logging.info(' Validation_loss : {}'.format(mean_validation_loss_value))
                    logging.info(' accuracy : {:0.2f}%'.format(mean_validation_accuracy_value * 100))
                    logging.info(' Percentage of pixels not background : {:0.2f}%'.format(percentage_pixels_not_bgr))
                    logging.info(' Percentage of pixels not background (GT) : {:0.2f}%'.format(percentage_pixels_not_bgr_gt))
                    train_writer.flush()

                # Save the weight of the network
                if step % FLAGS.save_steps_interval == 0 and step != start_step:
                    save_path = saver.save(sess, checkpoints_dir_save + "/model.ckpt", global_step=step)
                    logging.info("Model saved in file: %s" % save_path)

                step += 1

        except KeyboardInterrupt:
            logging.info('Interrupted')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            # Output computed metrics
            logging.info('-----------Step %d:-------------' % step)
            logging.info(' Final loss: {}'.format(G_final_loss_value))

            # Compute validation metrics
            sess.run(tf.local_variables_initializer())
            sess.run([iterator.initializer])
            n = 0
            local_confusion_matrix = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.int)
            non_zero_pixels_pred = non_zero_pixels_gt = total_pixels = 0

            for i in range(FLAGS.validation_batch_size):
                try:
                    cm, _, _, _, pred_img_value, gt_value = sess.run([confusion_matrix, mean_validation_update_op,
                                                    mean_IoU_update_op, accuracy_update_op, pred_img, gt])
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
                                              feed_dict={percentage_pixels_not_bgr_: percentage_pixels_not_bgr}), step)
            mIoU_no_bgr, mIoU_new_classes = tensor_utils.compute_and_print_IoU_per_class(local_confusion_matrix,
                            FLAGS.num_classes, FLAGS.is_incremental, FLAGS.from_new_class, FLAGS.to_new_class)
            train_writer.add_summary(sess.run(summary_mIoU_no_bgr, feed_dict={mIoU_no_bgr_: mIoU_no_bgr}), step)
            train_writer.add_summary(
                sess.run(summary_mIoU_new_classes, feed_dict={mIoU_new_classes_: mIoU_new_classes}), step)

            logging.info(' Validation_loss : {}'.format(mean_validation_loss_value))
            logging.info(' accuracy : {:0.2f}%'.format(mean_validation_accuracy_value * 100))
            logging.info(' Percentage of pixels not background : {:0.2f}%'.format(percentage_pixels_not_bgr))
            logging.info(' Percentage of pixels not background (GT) : {:0.2f}%'.format(percentage_pixels_not_bgr_gt))
            train_writer.flush()

            save_path = saver.save(sess, checkpoints_dir_save + "/model.ckpt", global_step=step)
            logging.info("Model saved in file: %s" % save_path)

            # When done, ask the threads to stop.
            coord.request_stop()
            coord.join(threads)

def main(unused_argv):
    train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
