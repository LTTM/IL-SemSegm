#######################################################################################
# Michieli U., Zanuttigh P.                                                           #
# "Incremental Learning Techniques for Semantic Segmentation"                         #
# Proceedings of the International Conference on Computer Vision (ICCV)               #
# Workshop on Transferring and Adapting Source Knowledge in Computer Vision (TASK-CV) #
# Seoul (South Korea), 2019.                                                          #
#######################################################################################

""" Freeze variables and convert the segmentation networks to a GraphDef file.
This makes file size smaller and can be used for inference in production.
An example of command-line usage is:
python export_graph.py --checkpoint_dir checkpoints/20170424-1152 \
                       --XtoY_model apple2orange.pb \
                       --YtoX_model orange2apple.pb \
                       --image_size 256
"""
import os
import tensorflow as tf
from model_with_distillation_blockencoder import model
from utils import image_utils

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('checkpoint_dir', './checkpoints/1_to_15', 'checkpoints directory path')
tf.flags.DEFINE_string('ckpt_load', '21000', 'checkpoints directory path')
tf.flags.DEFINE_string('model', 'model.pb', 'model name')

# Utilized data
tf.flags.DEFINE_string('training_file', '../../data/tfrecords/VOC2012/train_n_divided_2.tfrecords',
                       'tfrecords file for training')
tf.flags.DEFINE_string('validation_file', '../../data/tfrecords/VOC2012/val.tfrecords', 'tfrecords file for validation')
tf.flags.DEFINE_string('pretrained_folder', '../../data/pretrained', 'path of the folder for pretraied encoder models')

# Data format
tf.flags.DEFINE_integer('batch_size', 1, 'batch size')
tf.flags.DEFINE_integer('validation_batch_size', 1, ' validation batch size default: 1449')
tf.flags.DEFINE_integer('image_size', 321, 'image size, default: 256')
tf.flags.DEFINE_integer('num_classes', 21, 'number of output classes')
tf.flags.DEFINE_string('norm', 'batch', '[instance, batch] use instance norm or batch norm, default: instance')

# Training parameters
tf.flags.DEFINE_integer('training_steps', 20006, 'number of training steps')
tf.flags.DEFINE_integer('logging_steps_interval', 1, 'interval of steps between each log')
tf.flags.DEFINE_integer('validation_steps_interval', 1, 'interval of steps between each validation')
tf.flags.DEFINE_integer('save_steps_interval', 10000, 'interval of steps between each log')

# Checkpoint folder
tf.flags.DEFINE_string('load_model', '20181203-1525',
                       'folder of saved model that you wish to continue training (checkpoint/{})')

# Network optimization parameter (segmentation network uses SGD optimizer)
tf.flags.DEFINE_float('start_learning_rate', 2.5e-4, 'initial learning rate for segmentation network')
tf.flags.DEFINE_float('end_learning_rate', 0.0, 'final decay rate for segmentation network optimizer')
tf.flags.DEFINE_integer('start_decay_step', 0, 'number of steps after which start decay for segmentation network')
tf.flags.DEFINE_integer('decay_steps', 20000, 'number of steps of decay learning rate for segmentation network')
tf.flags.DEFINE_float('decay_power', 0.9, 'power of weight decay for segmentation network optimizer')
tf.flags.DEFINE_float('momentum', 0.9, 'momentum for segmentation network optimizer')
tf.flags.DEFINE_float('weight_decay', 1e-4, 'weight decay for momentum of segmentation network optimizer')

tf.flags.DEFINE_float('lambda_distillation', None, 'weight of the distillation loss. Expected: 0<=lambda_distillation')
tf.flags.DEFINE_integer('is_incremental', None, 'determine whether we are training in incremental way [1] or not [0]')
tf.flags.DEFINE_integer('is_save_new_softmax_maps', 0, 'determine whether to save new softmax maps [1] or not [0]')
tf.flags.DEFINE_integer('from_new_class', None, 'specifies the starting index [included] of the new classes to the added')
tf.flags.DEFINE_integer('to_new_class', None, 'specifies the ending index [included] of the new classes to be added')
tf.flags.DEFINE_string('standard_loss_applied_to', 'all', 'specifies where to apply the standard loss: [all, new, new_old] classes')
tf.flags.DEFINE_string('distill_loss_applied_to', 'all', 'specifies where to apply the distillation loss: [all, new, old] classes')

def export_graph(model_name):
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

        image_data = tf.placeholder(tf.string, name='input_image')
        input_image = tf.image.decode_jpeg(image_data, channels=3)
        input_image = tf.image.resize_images(input_image, size=(FLAGS.image_size, FLAGS.image_size))
        input_image = image_utils.convert2float(input_image)
        input_image.set_shape([FLAGS.image_size, FLAGS.image_size, 3])
        output_image = gan.G.sample(tf.expand_dims(input_image, 0))

        output_image = tf.identity(output_image, name='output_image')
        restore_saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        latest_ckpt = FLAGS.checkpoint_dir + '/model.ckpt-' + FLAGS.ckpt_load
        latest_ckpt_original = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        restore_saver.restore(sess, latest_ckpt)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, graph.as_graph_def(), [output_image.op.name])

        tf.train.write_graph(output_graph_def, FLAGS.checkpoint_dir+'/', model_name, as_text=False)

def main(unused_argv):
    print('Export model...')
    export_graph(FLAGS.model)


if __name__ == '__main__':
    tf.app.run()
