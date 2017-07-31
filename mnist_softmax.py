# -*- coding: utf-8 -*-

# softmax 分类器模型
import os
import argparse
import sys
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load dataset
from tensorflow.examples.tutorials.mnist import input_data


# define main()
def main(_):
    with tf.Graph().as_default():
        with tf.name_scope('Input'):
            X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
            Y_true = tf.placeholder(tf.float32, [None, 10], name='Y_true')

            # inference 前向预测
        with tf.name_scope('Inference'):
            # 模型参数变量
            w = tf.Variable(tf.zeros([784, 10]), name='weight')
            b = tf.Variable(tf.zeros([10]), name='bias')
            # inference y = wx + b
            logits = tf.add(tf.matmul(X, w), b)
            # softmax把Y_pred变成概率分布
            with tf.name_scope('Softmax'):
                Y_pred = tf.nn.softmax(logits=logits)
        # 定义损失
        with tf.name_scope('Loss'):
            trainLoss = tf.reduce_mean(-tf.reduce_sum(Y_true * tf.log(Y_pred), axis=1))
        # 定义损失
        with tf.name_scope('Train'):
            # 创建优化器
            Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            TrainOp = Optimizer.minimize(trainLoss)
        # 评估节点
        with tf.name_scope('Evaluate'):
            correct_predict = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_true, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        # 创建初始化
        init = tf.global_variables_initializer()
        # 保存计算图
        writer = tf.summary.FileWriter(logdir='logs/mnist_softmax1', graph=tf.get_default_graph())
        writer.close()
        # 使用数据集
        mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
        # 创建会话
        sess = tf.InteractiveSession()
        sess.run(init)

        # 递归执行
    for step in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        _, train_loss = sess.run([TrainOp, trainLoss], feed_dict={X: batch_xs, Y_true: batch_ys})
        # 打印输出
        print("train step", step, "train_loss=", "{:.9f}".format(train_loss))

    acc_score = sess.run(accuracy, feed_dict={X: mnist.test.images, Y_true: mnist.test.labels})
    print("预测准确率为：", acc_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='mnist_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
