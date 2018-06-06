# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 1.load data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# construct model
## 占位符(训练数据为 28*28=784 维)
input_x = tf.placeholder(tf.float32, [None, 784])
input_y = tf.placeholder(tf.float32, [None, 10])

## 模型的学习参数
weights = tf.get_variable('weights', [784, 10], initializer=tf.truncated_normal_initializer)
# weights = tf.Variable(tf.random_normal(([784, 10])))
biases = tf.Variable(tf.zeros([10]))

## 正向传播
logit = tf.matmul(input_x, weights)+biases

## 损失函数及优化函数
y_scores = tf.nn.softmax(logit)
loss = tf.reduce_mean(-tf.reduce_sum(input_y*tf.log(tf.clip_by_value(y_scores, 1e-30, 1.0)), 1))
global_step = tf.Variable(0, trainable=False)
optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss, global_step=global_step)

# 迭代训练
batch_size = 500
training_step = 200
display_step = 5

# 启动会话窗口
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(training_step):
        train_x, train_y = mnist.train.next_batch(batch_size)
        _, step, train_loss = sess.run([optimizer, global_step, loss], feed_dict={input_x: train_x, input_y: train_y})

        if step % display_step == 0:
            print('After {} epochs, loss on training set is {}'.format(step,train_loss))