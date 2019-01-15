import tensorflow as tf
import os
import numpy as np
import input_data

init_lr = 0.5
DEVICE_ID = "/gpu:0"
datasetPath = "Pima-training-set.txt"
testsetPath = "Pima-prediction-set.txt"
traindata = input_data.read_dataset(datasetPath)
testdata = input_data.read_dataset(testsetPath)
with tf.device(DEVICE_ID):
    config = tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 1
    
    x = tf.placeholder(tf.float32, [None, 8])
    W = tf.Variable(tf.zeros([8, 16]))
    b = tf.Variable(tf.zeros([16]))
    h1 = tf.matmul(x, W)+b
    W_2 = tf.Variable(tf.zeros([16, 2]))
    b_2 = tf.Variable(tf.zeros([2]))
    y = tf.nn.softmax(tf.matmul(h1, W_2) + b_2)

    y_ = tf.placeholder(tf.float32, [None, 2])
    cur_lr = tf.placeholder("float")
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(cur_lr).minimize(cross_entropy)
    init = tf.initialize_all_variables()

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(init)

    # print traindata.next_batch(8);exit()
    test_batchx , test_batchy = testdata.next_batch(160)
    for i in range(1000000):
        batch_x, batch_y = traindata.next_batch(32)
        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, cur_lr:init_lr})
        if i % 5000 == 0:
            accu = sess.run(accuracy, feed_dict={x: test_batchx, y_: test_batchy, cur_lr:init_lr})
            print "iter", i, "Accuracy in test set:", accu
        if i % 100000 == 0:
            init_lr = init_lr / 10
