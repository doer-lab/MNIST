# coding: utf-8

# 导入相关函数库
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# CNN 前向传播参数
num_channels = 1

conv1_size = 3
conv1_deep = 32

conv2_size = 3
conv2_deep = 64

fc1_nodes = 128
num_classes = 10

# CNN 前向传播过程
def cnn_inference(input_x):
    """ conv -> pooling -> conv -> pooling -> full connection -> full connection """
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weights',[conv1_size, conv1_size, num_channels, conv1_deep],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('biases',[conv1_deep],initializer=tf.zeros_initializer)
        conv1 = tf.nn.conv2d(input_x,conv1_weights,strides=[1,2,2,1],padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
    
    with tf.variable_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weights',[conv2_size,conv2_size,conv1_deep,conv2_deep],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('biases',[conv2_deep],initializer=tf.zeros_initializer)
        conv2 = tf.nn.conv2d(pool1,conv2_weights,strides=[1,2,2,1],padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))
    
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
    pool2_shape = tf.shape(pool2)
    # from functools import reduce
    # total_nodes = reduce(tf.multiply,pool2_shape[1:])
    total_nodes = pool2_shape[1]*pool2_shape[2]*pool2_shape[3]
    pool2_flat = tf.reshape(pool2,(-1,total_nodes))

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weights',[total_nodes,fc1_nodes],initializer=tf.truncated_normal_initializer)
        fc1_biases = tf.get_variable('biases',[fc1_nodes],initializer=tf.zeros_initializer)
        fc1 = tf.multiply(pool2_flat,fc1_weights)+fc1_biases
        relu3 = tf.nn.relu(fc1)
    
    with tf.variable_scope('layer6-output'):
        output_weights = tf.get_variable('weights',[fc1_nodes,num_classes],initializer=tf.truncated_normal_initializer)
        output_biases = tf.get_variable('biases',[num_classes],initializer=tf.zeros_initializer)
        logit = tf.multiply(relu3,output_weights)+output_biases
    
    return logit


# define a function to reshape raw data
def data_reshape(data):
    return np.reshape(data,[-1,28,28,1])

# load data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

# define placeholder
x = tf.placeholder(tf.float32,[None,28,28,1])
y = tf.placeholder(tf.float32,[None,10])

# inference
y_ = cnn_inference(x)

# define loss function
scores = tf.nn.softmax(y_)
cross_entropy = -tf.reduce_sum(y*tf.log(tf.clip_by_value(scores,1e-30,1.0)),1)
loss = tf.reduce_mean(cross_entropy)

learning_rate = 0.001
global_step = tf.Variable(1,trainable=False)
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)

# initializing all variables and trainging model
batch_size = 300
training_epochs = 3000
display_epoch = 10

with  tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        train_x, train_y = mnist.train.next_batch(batch_size)
        _,training_loss = sess.run([optimizer,loss],feed_dict={x:data_reshape(train_x),y:train_y})

        if (epoch+1) % display_epoch == 0:
            print('After {} epochs, loss on training data is {}'.format(epoch+1))
