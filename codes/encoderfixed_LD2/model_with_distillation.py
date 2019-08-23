#######################################################################################
# Michieli U., Zanuttigh P.                                                           #
# "Incremental Learning Techniques for Semantic Segmentation"                         #
# Proceedings of the International Conference on Computer Vision (ICCV)               #
# Workshop on Transferring and Adapting Source Knowledge in Computer Vision (TASK-CV) #
# Seoul (South Korea), 2019.                                                          #
#######################################################################################

import tensorflow as tf
from network_blockencoder import Network
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
                 lambda_distillation,
                 is_incremental,
                 from_new_class,
                 to_new_class,
                 standard_loss_applied_to,
                 distill_loss_applied_to,
                 is_distillonfeatures,
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
        self.lambda_distillation = lambda_distillation
        self.tau = 1
        self.network_input = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 3])
        self.network_input_labels = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 1])
        self.network_input_output_maps = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, num_classes])
        self.network_input_feature_maps = tf.placeholder(tf.float32, [self.batch_size, 41, 41, 2048])
        self.is_incremental = is_incremental
        self.from_new_class = from_new_class
        self.to_new_class = to_new_class
        self.standard_loss_applied_to = standard_loss_applied_to
        self.distill_loss_applied_to = distill_loss_applied_to
        self.is_distillonfeatures = is_distillonfeatures


        # Build network structure
        self.G = Network(name='SegmentNetwork', image_size=image_size, num_classes=num_classes,
                         pretrained_folder=pretrained_folder, weight_decay=self.weight_decay)

    def training_model(self):
        """
        Build the tensorflow graph

        :return: parameters needed to perform the training
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


        # Initialize variables
        G_loss = 0
        distill_loss = []
        final_loss = 0

        training_summary_list = []
        G_loss_summary = None
        G_optimizer = None
        G_optim_summary = None
        G_new_softmax = None
        G_new_features = None

        # Define loss functions
        if self.is_incremental:
            G_loss, G_new_softmax, G_new_features, G_loss_summary = self.network_loss(G_input=self.network_input,
                                                                            gt_labels=self.network_input_labels,
                                                                            applied_to=self.standard_loss_applied_to)
        else:
            # Define readers
            reader = Reader(self.train_file, name='reader_train', image_size=self.image_size,
                            batch_size=self.batch_size, min_queue_examples=1000, seed=12345)
            # Read input and output data from dataset
            G_input, G_labels, G_reader_summary = reader.feed()
            training_summary_list.append(G_reader_summary)
            # Define loss functions
            G_loss, _, _, G_loss_summary = self.network_loss(G_input=G_input, gt_labels=G_labels)

        if not self.lambda_distillation == 0:
            if not self.is_distillonfeatures:  # distillation applied on the output
                distill_loss, distill_loss_summary = self.distillation_loss(new_softmax=G_new_softmax,
                                                                            old_softmax=self.network_input_output_maps,
                                                                            gt_labels = self.network_input_labels,
                                                                            applied_to=self.distill_loss_applied_to)
            else: # distillation applied to the feature space
                distill_loss, distill_loss_summary = self.distillation_loss_features(new_features=G_new_features,
                                                                            old_features=self.network_input_feature_maps)

            G_new_softmax = tf.nn.softmax(G_new_softmax)
            training_summary_list.append(distill_loss_summary)

            final_loss = G_loss + self.lambda_distillation*distill_loss
        else:
            final_loss = G_loss

        # Define optimizers
        G_optimizer, G_optim_summary = self.network_optimizer(name='G_optimizer', loss=final_loss)

        # Define summary
        training_summary_list.append(G_loss_summary)
        training_summary_list.append(G_optim_summary)

        # training_summary_list.append(G_reader_summary)
        training_summary = tf.summary.merge(training_summary_list)

        return G_loss, distill_loss, final_loss, G_optimizer, training_summary, G_new_softmax, G_new_features

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
        val_image = next_val_data[0]
        val_label = next_val_data[1]

        # Validation loss
        validation_loss, output_maps = self.compute_validation_loss(G_input=val_image, gt_labels=val_label)
        mean_validation_loss, mean_validation_update_op = tf.metrics.mean(validation_loss)

        # Mean IoU and pixel accuracy
        pred_img, gt, mean_IoU, mean_IoU_update_op, accuracy, accuracy_update_op, confusion_matrix = \
            self.compute_evaluation_metrics(G_input=val_image, gt_labels=val_label)

        # Summary
        validation_summary_list = [tf.summary.scalar('validation/validation_loss', mean_validation_loss),
                                   tf.summary.scalar('validation/mean_IoU', mean_IoU),
                                   tf.summary.scalar('validation/pixel_accuracy', accuracy)]
        validation_summary = tf.summary.merge(validation_summary_list)

        return pred_img, gt, iterator, mean_validation_loss, mean_validation_update_op, confusion_matrix, accuracy, \
               accuracy_update_op, mean_IoU, mean_IoU_update_op, output_maps, validation_summary

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

    def network_loss(self, G_input, gt_labels, applied_to='all'):
        """
        Compute the loss of the segmentation network (supervised)

        :param G_input: input of the segmentation network. 4D tensor: [batch_size, image_width, image_height, 3]
        :param gt_labels: gt with classes. 4D tensor: [batch_size, image_width, image_height, 1]
        :return: (supervised) loss of the segmentation network
        """
        # Compute needed variables
        G_output, G_features = self.G(G_input)
        gt_one_hot = tensor_utils.convert_label2onehot(gt_labels, self.num_classes)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=G_output, labels=gt_one_hot)

        # Compute the cross entropy loss
        if applied_to == 'all':
            target_mask = tf.squeeze(tf.less_equal(gt_labels, self.num_classes - 1), axis=3)
            # considers only the classes from 0 to num_classes-1 (this has to be done also in all the following masks)
            loss_masked = tf.boolean_mask(loss, target_mask)
        if applied_to == 'new':
            # compute only the loss for the new classes we are adding ([from_new_class, to_new_class])
            target_mask = tf.squeeze(tf.logical_and(tf.logical_and(tf.greater_equal(gt_labels, self.from_new_class),
                                                                   tf.less_equal(gt_labels, self.to_new_class)),
                                                    tf.less_equal(gt_labels, self.num_classes - 1)), axis=3)
            loss_masked = tf.boolean_mask(loss, target_mask)
        elif applied_to == 'new_old':
            # compute the loss for the new and the old classes, background excluded ([1, to_new_class])
            target_mask = tf.squeeze(tf.logical_and(tf.logical_and(tf.greater_equal(gt_labels, 1),
                                                                   tf.less_equal(gt_labels, self.to_new_class)),
                                                    tf.less_equal(gt_labels, self.num_classes - 1)), axis=3)
            loss_masked = tf.boolean_mask(loss, target_mask)

        loss = tf.reduce_mean(loss_masked)

        # Summaries
        image_slice_pred = image_utils.convert_output2rgb(
            tf.slice(G_output, [0, 0, 0, 0], [1, self.image_size, self.image_size, self.num_classes]))
        image_slice_gt = image_utils.convert_output2rgb(
            tf.slice(gt_one_hot, [0, 0, 0, 0], [1, self.image_size, self.image_size, self.num_classes]))

        summary = [tf.summary.scalar('training_losses/standard_loss', loss),
                   tf.summary.image('output_image/predicted_image', image_slice_pred),
                   tf.summary.image('output_image/GT_image', image_slice_gt)]
        return loss, G_output, G_features, summary

    def distillation_loss(self, new_softmax, old_softmax, gt_labels, applied_to='all'):
        """
        Compute the distillation loss between the previous softmax and the current one

        :param new_softmax: output of the softmax of the current network. 4D tensor: [batch_size, image_width, image_height, 21]
        It may have some zeros at the end of the tensor
        :param old_softmax: output of the softmax of the current network. 4D tensor: [batch_size, image_width, image_height, 21]
        It may have some zeros at the end of the tensor
        :return: distillation loss
        """

        # Compute the cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=new_softmax, labels=old_softmax)

        if applied_to == 'all':
            target_mask = tf.squeeze(tf.less_equal(gt_labels, self.num_classes - 1), axis=3)
            loss_masked = tf.boolean_mask(loss, target_mask)
        if applied_to == 'old':
            # compute only the loss for the old classes ([1, from_new_class-1])
            target_mask = tf.squeeze(tf.logical_and(tf.greater_equal(gt_labels, 1),
                                                    tf.less_equal(gt_labels, self.from_new_class-1)),axis=3)
            loss_masked = tf.boolean_mask(loss, target_mask)
        elif applied_to == 'old_bgr':
            # compute the loss for the old classes and the background included ([0,from_new_class-1])
            target_mask = tf.squeeze(tf.less_equal(gt_labels, self.from_new_class - 1), axis=3)
            loss_masked = tf.boolean_mask(loss, target_mask)

        loss = tf.reduce_mean(loss_masked)
        # The distillation loss would be negative if applied_to='old' and the ground_truth does not contain old_classes
        # to retain. Hence we need to change from nan to 0.
        # In principle this could also happen in the case applied_to='old_bgr', but this is unlikely since the
        # background class is usually present in all the images.
        loss = tf.where(tf.is_nan(loss), 0., loss)

        # Summary
        summary = [tf.summary.scalar('training_losses/distillation_loss', loss)]
        return loss, summary

    def distillation_loss_features(self, new_features, old_features):
        """
        Compute the distillation loss between the previous feature maps and the current ones

        :param new_features: output of the feature maps of the current network. 4D tensor: [batch_size, 41, 41, 2048]
        :param old_features: output of the feature maps of the previous network. 4D tensor: [batch_size, 41, 41, 2048]
        :return: distillation loss on the feature space
        """

        # Compute the cross entropy loss
        loss = tf.losses.mean_squared_error(labels=old_features, predictions=new_features)
        #loss = tf.reduce_mean(loss)

        # Summary
        summary = [tf.summary.scalar('training_losses/distillation_loss_features', loss)]
        return loss, summary

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
        loss, output_maps, _, _ = self.network_loss(G_input=G_input, gt_labels=gt_labels)

        return loss, output_maps

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
        pred_reshaped = tf.reshape(pred, [-1, ])
        gt_reshaped = tf.reshape(gt_labels, [-1, ])

        # Ignoring all labels greater than or equal to n_classes.
        mask = tf.less_equal(gt_reshaped, self.num_classes - 1)
        pred = tf.boolean_mask(pred_reshaped, mask)
        gt = tf.boolean_mask(gt_reshaped, mask)

        # mIoU
        mIoU, mIou_update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=self.num_classes)

        # confusion matrix
        confusion_matrix = tf.confusion_matrix(predictions=pred, labels=gt, num_classes=self.num_classes)

        # Pixel accuracy
        accuracy, accuracy_update_op = tf.metrics.accuracy(predictions=pred, labels=gt)

        return pred, gt, mIoU, mIou_update_op, accuracy, accuracy_update_op, confusion_matrix
