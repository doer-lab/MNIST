# coding: utf-8

# load relevant packages
import tensorflow as tf

# parameters
image_size = 28
num_channels = 1

conv1_size = 3
conv1_deep = 32

pool1_size = 2

conv2_size = 3
conv2_deep = 64

pool2_size = 2

fc1_node = 512

num_classes = 10

# build a model by the way of class
# model structure: input -> convolutional -> max pooliing -> convolutional -> max pooling -> full connection -> full connection -> output

# # version 1: general model
# class ImageCnn():
#     """ define a CNN for images classification """
#     def __init__(self):
#         """
#          initializing class of ImageCnn
#          input_x has shape [None, 28, 28, 1]
#          input_y has shape [None, 10]
#         """
#         self.input_x = tf.placeholder(tf.float32, [None, image_size, image_size, num_channels], name='input_x')
#         self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')

#         # first layer: convolutional layer
#         with tf.variable_scope('layer1-conv1'):
#             """ filter size is 3x3 """
#             conv1_weights = tf.get_variable('weights', [conv1_size, conv1_size, num_channels, conv1_deep],
#                                             initializer=tf.truncated_normal_initializer(stddev=0.1))
#             conv1_biases = tf.get_variable('biases', [conv1_deep], initializer=tf.zeros_initializer)

#             conv1 = tf.nn.conv2d(self.input_x,
#                                  conv1_weights,
#                                  strides=[1, 2, 2, 1],
#                                  padding='SAME')
#             relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

#         # second layer: pooling layer
#         with tf.variable_scope('layer2-pool1'):
#             pool1 = tf.nn.max_pool(relu1,
#                                    ksize=[1, 2, 2, 1],
#                                    strides=[1, 2, 2, 1],
#                                    padding='SAME')

#         # third layer: convolutional layer
#         with tf.variable_scope('layer3-conv2'):
#             conv2_weights = tf.get_variable('weights', [conv2_size, conv2_size, conv1_deep, conv2_deep],
#                                             initializer=tf.truncated_normal_initializer(stddev=0.1))
#             conv2_biases = tf.get_variable('biases', [conv2_deep], initializer=tf.constant_initializer(0.1))
#             conv2 = tf.nn.conv2d(pool1,
#                                  conv2_weights,
#                                  strides=[1, 2, 2, 1],
#                                  padding='SAME')
#             relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

#         # fourth layer: pooling layer
#         with tf.variable_scope('layer4-pool2'):
#             pool2 = tf.nn.max_pool(relu2,
#                                    ksize=[1, 2, 2, 1],
#                                    strides=[1, 2, 2, 1],
#                                    padding='SAME')

#         # reshape pool2 first
#         pool2_shape = pool2.get_shape().as_list()            # pool2 shape is [None, , , conv2_deep], results is a tensor by using TensorFlow, so, need to covert tensor to a list
#         total_nodes = pool2_shape[1]*pool2_shape[2]*pool2_shape[3]
#         pool2_reshape = tf.reshape(pool2, [-1, total_nodes])

#         # fifth layer: full connection layer
#         with tf.variable_scope('layer5-fc1'):
#             fc_weights = tf.get_variable('weights', [total_nodes, fc1_node],
#                                          initializer=tf.random_normal_initializer)
#             fc1_biases = tf.get_variable('biases', [fc1_node], initializer=tf.constant_initializer(0.1))
#             fc1 = tf.nn.relu(tf.matmul(pool2_reshape, fc_weights)+fc1_biases)

#         # sixth layer: full connection layer
#         with tf.variable_scope('layer6-fc2'):
#             fc2_weights = tf.get_variable('weights', [fc1_node, num_classes],
#                                           initializer=tf.truncated_normal_initializer)
#             fc2_biases = tf.get_variable('biases', [num_channels],
#                                          initializer=tf.constant_initializer(0.5))
#             logit = tf.matmul(fc1, fc2_weights)+fc2_biases

#         # compute prediction value
#         with tf.variable_scope('loss'):
#             y_prob = tf.nn.softmax(logit)
#             cross_entropy = -tf.reduce_sum(self.input_y*tf.log(tf.clip_by_value(y_prob, 1e-30, 1)), reduction_indices=1)
#             self.loss = tf.reduce_mean(cross_entropy)

#         # compute accuracy
#         with tf.variable_scope('accuracy'):
#             correct_predictions = tf.equal(tf.argmax(self.input_y, 1), tf.argmax(y_prob, 1))
#             self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


# # version 2: with dropout
class ImageCnn():
    """ define a CNN for images classification """
    def __init__(self):
        """
         initializing class of ImageCnn
         input_x has shape [None, 28, 28, 1]
         input_y has shape [None, 10]
        """
        self.input_x = tf.placeholder(tf.float32, [None, image_size, image_size, num_channels], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')

        # first layer: convolutional layer
        with tf.variable_scope('layer1-conv1'):
            """ filter size is 3x3 """
            conv1_weights = tf.get_variable('weights', [conv1_size, conv1_size, num_channels, conv1_deep],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable('biases', [conv1_deep], initializer=tf.zeros_initializer)

            conv1 = tf.nn.conv2d(self.input_x,
                                 conv1_weights,
                                 strides=[1, 2, 2, 1],
                                 padding='SAME')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

        # second layer: pooling layer
        with tf.variable_scope('layer2-pool1'):
            pool1 = tf.nn.max_pool(relu1,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

        # third layer: convolutional layer
        with tf.variable_scope('layer3-conv2'):
            conv2_weights = tf.get_variable('weights', [conv2_size, conv2_size, conv1_deep, conv2_deep],
                                            initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable('biases', [conv2_deep], initializer=tf.constant_initializer(0.1))
            conv2 = tf.nn.conv2d(pool1,
                                 conv2_weights,
                                 strides=[1, 2, 2, 1],
                                 padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

        # fourth layer: pooling layer
        with tf.variable_scope('layer4-pool2'):
            pool2 = tf.nn.max_pool(relu2,
                                   ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

        # reshape pool2 first
        pool2_shape = pool2.get_shape().as_list()            # pool2 shape is [None, , , conv2_deep], results is a tensor by using TensorFlow, so, need to covert tensor to a list
        total_nodes = pool2_shape[1]*pool2_shape[2]*pool2_shape[3]
        pool2_reshape = tf.reshape(pool2, [-1, total_nodes])

        # fifth layer: full connection layer
        with tf.variable_scope('layer5-fc1'):
            fc_weights = tf.get_variable('weights', [total_nodes, fc1_node],
                                         initializer=tf.random_normal_initializer)
            fc1_biases = tf.get_variable('biases', [fc1_node], initializer=tf.constant_initializer(0.1))
            fc1 = tf.nn.relu(tf.matmul(pool2_reshape, fc_weights)+fc1_biases)

        # sixth layer: dropout layer
        with tf.variable_scope('layer6-dropout'):
            fc1_dropout = tf.nn.dropout(fc1,keep_prob=self.dropout_keep_prob)

        # seventh layer: full connection layer
        with tf.variable_scope('layer7-fc2'):
            fc2_weights = tf.get_variable('weights', [fc1_node, num_classes],
                                          initializer=tf.truncated_normal_initializer)
            fc2_biases = tf.get_variable('biases', [num_channels],
                                         initializer=tf.constant_initializer(0.5))
            logit = tf.matmul(fc1_dropout, fc2_weights)+fc2_biases

        # compute prediction value
        with tf.variable_scope('loss'):
            y_prob = tf.nn.softmax(logit)
            cross_entropy = -tf.reduce_sum(self.input_y*tf.log(tf.clip_by_value(y_prob, 1e-30, 1)), reduction_indices=1)
            self.loss = tf.reduce_mean(cross_entropy)

        # compute accuracy
        with tf.variable_scope('accuracy'):
            correct_predictions = tf.equal(tf.argmax(self.input_y, 1), tf.argmax(y_prob, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
