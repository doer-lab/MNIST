# coding: utf-8

import tensorflow as tf
from class_cnn import *
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
batch_size = 100
training_epochs = 2000
display_step = 2
#
# imagecnn = ImageCnn()
#
# # load data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
# # define optimizer
# learning_rate = 0.001
# global_step = tf.Variable(1, trainable=False)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(imagecnn.loss, global_step=global_step)
#
# # iteration
# training_epochs = 200
# display_step = 10
# batch_size = 200
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     for epoch in range(training_epochs):
#         train_x, train_y = mnist.train.next_batch(batch_size=batch_size)
#         train_x_reshaped = tf.reshape(train_x, (-1, 28, 28, 1))
#         _,step,loss = sess.run([optimizer,global_step,imagecnn.loss],feed_dict={imagecnn.input_x:train_x_reshaped,imagecnn.input_y:train_y})



def train(mnist):

    with tf.Session() as sess:
        imagecnn = ImageCnn()
        global_step = tf.Variable(0, trainable=False)
        optimizer = tf.train.AdadeltaOptimizer(0.001).minimize(imagecnn.loss, global_step=global_step)

        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epochs):
            train_x, train_y = mnist.train.next_batch(batch_size)
            train_x_shaped = np.reshape(train_x, [-1, 28, 28, 1])
            feed_train = {imagecnn.input_x: train_x_shaped, imagecnn.input_y: train_y}
            _, step, loss = sess.run([optimizer, global_step, imagecnn.loss], feed_dict=feed_train)

            if epoch % display_step == 0:
                print('After {} steps, loss on training data is {}'.format(step, loss))

# load data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
train(mnist)