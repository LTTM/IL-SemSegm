#######################################################################################
# Michieli U., Zanuttigh P.                                                           #
# "Incremental Learning Techniques for Semantic Segmentation"                         #
# Proceedings of the International Conference on Computer Vision (ICCV)               #
# Workshop on Transferring and Adapting Source Knowledge in Computer Vision (TASK-CV) #
# Seoul (South Korea), 2019.                                                          #
#######################################################################################

import tensorflow as tf

from network import Network
from reader import Reader
from utils import image_utils, tensor_utils

REAL_LABEL = 0.9


class model:
    def __init__(self,
                 train_file,
                 validation_file,
                 batch_size,
                 image_size,
                 num_classes,
                 pretrained_folder,
                 start_learning_rate,
                 momentum,
                 weight_decay,
                 decay_power,
                 decay_steps,
                 start_decay_step,
                 end_learning_rate,
                 ):

        self.train_file = train_file
        self.validation_file = validation_file
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_classes = num_classes
        self.pretrained_folder = pretrained_folder
        self.start_learning_rate = start_learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.decay_power = decay_power
        self.decay_steps = decay_steps
        self.start_decay_step = start_decay_step
        self.end_learning_rate = end_learning_rate
        self.tau = 1

        # Build network structure
        self.G = Network(name='SegmentNetwork', image_size=image_size, num_classes=num_classes,
                         pretrained_folder=pretrained_folder, weight_decay=self.weight_decay)

    def validation_model(self):
        """
        Build the tensorflow graph used to compute prediction scores on validation data

        :return: parameters needed to perform the validation
        """

        def _extract_fn(tfrecord):
            """
            Utility function used to extract data for validation

            :param tfrecord: file containing validation data
            :return: single data contained in the tfrecord file
            """
            # Extract features using the keys set during creation
            features = {
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'input': tf.FixedLenFeature([], tf.string),
                'gt': tf.FixedLenFeature([], tf.string)
            }

            # Obtain back image and labels
            sample = tf.parse_single_example(tfrecord, features)
            height = tf.cast(sample['height'], tf.int32)
            width = tf.cast(sample['width'], tf.int32)
            image = tf.image.decode_jpeg(sample['input'], channels=3)
            gt = tf.image.decode_png(sample['gt'], channels=1)

            # Preprocess paper_images
            image, gt = image_utils.preprocess_val_data(image, gt)
            return [image, gt]

        # Build iterator for the validation dataset
        validation_dataset = tf.data.TFRecordDataset(self.validation_file).map(_extract_fn)
        iterator = validation_dataset.make_initializable_iterator()
        next_val_data = iterator.get_next()

        # Obtain validation data
        val_image, val_label = image_utils.preprocess_train_data(next_val_data[0], next_val_data[1], self.image_size)

        # Validation loss
        validation_loss, output_maps, features_map = self.compute_validation_loss(G_input=val_image, gt_labels=val_label)
        output_maps = tf.nn.softmax(output_maps)
        mean_validation_loss, mean_validation_update_op = tf.metrics.mean(validation_loss)

        # Mean IoU
        mean_IoU, mean_IoU_update_op, accuracy, accuracy_update_op, confusion_matrix = self.compute_evaluation_metrics(
            G_input=val_image, gt_labels=val_label)

        # Summary
        validation_summary_list = [tf.summary.scalar('validation/validation_loss', mean_validation_loss),
                                   tf.summary.scalar('validation/mean_IoU', mean_IoU),
                                   tf.summary.scalar('validation/pixel_accuracy', accuracy)]
        validation_summary = tf.summary.merge(validation_summary_list)

        return iterator, mean_validation_loss, mean_validation_update_op, confusion_matrix, accuracy, \
               accuracy_update_op, mean_IoU, mean_IoU_update_op, output_maps, features_map, val_image, val_label, \
               validation_summary

    def network_optimizer(self, name, loss):
        """
        Define an optimizer for the segmentation network

        :param name: name of the optimizer
        :param loss: loss to optimize
        :return: optimizer
        """
        global_step = tf.Variable(0, trainable=False)

        learning_rate = (
            tf.where(
                tf.greater_equal(global_step, self.start_decay_step),
                tf.train.polynomial_decay(self.start_learning_rate, global_step - self.start_decay_step,
                                          self.decay_steps, self.end_learning_rate,
                                          power=self.decay_power),
                self.start_learning_rate
            )
        )
        summary = tf.summary.scalar('learning_rate/SGD_{}'.format(name), learning_rate)

        learning_step = (
            tf.train.MomentumOptimizer(learning_rate, momentum=self.momentum, name=name)
                .minimize(loss + self.G.l2_loss, global_step=global_step, var_list=self.G.all_var)
        )
        return learning_step, summary

    def network_loss(self, G_input, gt_labels):
        """
        Compute the loss of the segmentation network (supervised)

        :param G_input: input of the segmentation network. 4D tensor: [batch_size, image_width, image_height, 3]
        :param gt_labels: gt with classes. 4D tensor: [batch_size, image_width, image_height, 1]
        :return: (supervised) loss of the segmentation network
        """
        # Compute needed variables
        G_output, G_features = self.G(G_input)
        gt_one_hot = tensor_utils.convert_label2onehot(gt_labels, self.num_classes)

        # Compute the cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=G_output, labels=gt_one_hot)
        target_mask = tf.squeeze(tf.less_equal(gt_labels, self.num_classes - 1), axis=3)
        loss_masked = tf.boolean_mask(loss, target_mask)
        loss = tf.reduce_mean(loss_masked)

        # Summary
        image_slice = image_utils.convert_output2rgb(
            tf.slice(G_output, [0, 0, 0, 0], [1, self.image_size, self.image_size, self.num_classes]))
        summary = [tf.summary.scalar('loss_G/G', loss),
                   tf.summary.image('G_sup/output', image_slice)]

        return loss, G_output, G_features, summary

    def compute_validation_loss(self, G_input, gt_labels):
        """
        Compute the validation loss

        :param G_input: input of the segmentation network. 4D tensor: [batch_size, image_width, image_height, 3]
        :param gt_labels: gt with classes. 4D tensor: [batch_size, image_width, image_height, 1]
        :return: validation loss
        """
        # Reformat input values
        G_input = tf.expand_dims(G_input, axis=0)
        gt_labels = tf.expand_dims(gt_labels, axis=0)

        # Compute loss
        loss, output_maps, feature_maps, _ = self.network_loss(G_input=G_input, gt_labels=gt_labels)

        return loss, output_maps, feature_maps

    def compute_evaluation_metrics(self, G_input, gt_labels):
        """
        Compute all the metrics for the evaluation

        :param G_input: input of the segmentation network. 4D tensor: [batch_size, image_width, image_height, 3]
        :param gt_labels: gt with classes. 4D tensor: [batch_size, image_width, image_height, 1]
        :return:
                mIoU: mean intersection over union
                mIou_update_op: update operation for mIoU
                accuracy: pixel accuracy
                accuracy_update_op: update operation for pixel accuracy
        """
        # Reformat input values
        G_input = tf.expand_dims(G_input, axis=0)
        gt_labels = tf.expand_dims(gt_labels, axis=0)

        # predictions
        raw_output, _ = self.G(G_input)
        raw_output = tf.argmax(raw_output, axis=3)
        pred = tf.expand_dims(raw_output, axis=3)
        pred = tf.reshape(pred, [-1, ])
        gt = tf.reshape(gt_labels, [-1, ])

        # Ignoring all labels greater than or equal to n_classes.
        mask = tf.less_equal(gt, self.num_classes - 1)
        pred = tf.boolean_mask(pred, mask)
        gt = tf.boolean_mask(gt, mask)

        # mIoU
        mIoU, mIou_update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=self.num_classes)

        # confusion matrix
        confusion_matrix = tf.confusion_matrix(predictions=pred, labels=gt, num_classes=self.num_classes)

        # Pixel accuracy
        accuracy, accuracy_update_op = tf.metrics.accuracy(predictions=pred, labels=gt)

        return mIoU, mIou_update_op, accuracy, accuracy_update_op, confusion_matrix
