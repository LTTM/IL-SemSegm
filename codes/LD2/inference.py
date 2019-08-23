#######################################################################################
# Michieli U., Zanuttigh P.                                                           #
# "Incremental Learning Techniques for Semantic Segmentation"                         #
# Proceedings of the International Conference on Computer Vision (ICCV)               #
# Workshop on Transferring and Adapting Source Knowledge in Computer Vision (TASK-CV) #
# Seoul (South Korea), 2019.                                                          #
#######################################################################################

"""Translate an image to another image
An example of command-line usage is:
python export_graph.py --model pretrained/apple2orange.pb \
                       --input input_sample.jpg \
                       --output output_sample.jpg \
                       --image_size 256
"""
# Check output size. Now only works with 321x321 images, because the export_graph.py file freezes the graph with a
# specified size

import tensorflow as tf
import cv2

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('model', 'checkpoints/1_to_10_11_11_distill_lambda1.0_std_new_old_dist_all/model.pb', 'model path (.pb)')
tf.flags.DEFINE_string('input_folder', '../../../dataset/incremental_images/val_img_11_11_notin_following/', 'input image path (.jpg)')
tf.flags.DEFINE_string('GT_images', None, 'GT input image path')
tf.flags.DEFINE_string('output_folder', './output_images_val_11_notin_following/', 'output image path (.jpg)')
tf.flags.DEFINE_integer('image_size', '321', 'image size, default: 256')

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def main(unused_argv):
    from os import listdir, makedirs, error
    from os.path import isfile, join

    graph = load_graph(FLAGS.model)

    input_image = graph.get_tensor_by_name('prefix/input_image:0')
    output_image = graph.get_tensor_by_name('prefix/output_image:0')

    try:
        makedirs(FLAGS.output_folder)
    except error:
        pass

    files = [f for f in listdir(FLAGS.input_folder) if isfile(join(FLAGS.input_folder, f))]
    with tf.Session(graph=graph) as sess:
        for i in range(len(files)):
            image_name = files[i]

            image_data = tf.gfile.FastGFile(FLAGS.input_folder + image_name, 'rb').read()
            generated = output_image.eval(feed_dict={input_image: image_data})

            image_data = cv2.imread(FLAGS.input_folder + image_name[0:-4] + '.jpg')
            image_data_resized = cv2.resize(image_data, dsize=(FLAGS.image_size, FLAGS.image_size), interpolation=cv2.INTER_LINEAR)

            GT = cv2.imread(FLAGS.GT_images + image_name[0:-4] + '.png')
            GT_resized = cv2.resize(GT, dsize=(FLAGS.image_size, FLAGS.image_size), interpolation=cv2.INTER_NEAREST)

            with open(FLAGS.output_folder + image_name[0:-4] + '_predicted.png', 'wb') as f:
                f.write(generated)

            cv2.imwrite(FLAGS.output_folder + image_name[0:-4] + '_GT.png', GT_resized)
            cv2.imwrite(FLAGS.output_folder + image_name[0:-4] + '_RGB.png', image_data_resized)

            print('Elaborated file {}/{}.'.format(i + 1, len(files)))


if __name__ == '__main__':
    tf.app.run()
