import tensorflow as tf
import os
import numpy as np
import input_data

datasetPath = "Pima-training-set.txt"
testsetPath = "Pima-prediction-set.txt"
traindata = input_data.read_dataset(datasetPath)
testdata = input_data.read_dataset(testsetPath)

x = tf.placeholder(tf.float32, [None, 8])
W = tf.Variable(tf.zeros([8, 2]))
b = tf.Variable(tf.zeros([2]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(init)

# print traindata.next_batch(8);exit()
test_batchx , test_batchy = testdata.next_batch(160)
for i in range(100000):
    batch_x, batch_y = traindata.next_batch(32)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
    if i % 5000 == 0:
        print(sess.run(accuracy, feed_dict={x: test_batchx, y_: test_batchy}))
    
