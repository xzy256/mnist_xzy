# -*- coding: utf-8 -*-
"""
使用Feed字典训练和评估 MNIST 网络
"""

import argparse
import os.path
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import mnist

os.environ['TF_CPP_LOG_LEVEL'] = '2'

# 定义全局变量
FLAGS = None


# 定义占位，默认batch_size = 100
def placeholder_inputs(batch_size):
    # IMAGE_PIXELS = 28 * 28 pixels
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


# 填充数据
def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


# 做评估
def do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_set):
    true_count = 0
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size  # 取整操作
    num_examples = steps_per_epoch * FLAGS.batch_size
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training():
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)
        # 前向预测
        logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
        # 创建损失
        loss = mnist.loss(logits, labels_placeholder)
        # 训练
        train_op = mnist.training(loss, FLAGS.learning_rate)
        # 评估
        eval_correct = mnist.evaluation(logits, labels_placeholder)
        # 添加summary
        mergeSummary = tf.summary.merge_all()
        # 创建一个saver
        saver = tf.train.Saver()
        # 初始化
        init = tf.global_variables_initializer()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        summary_writer.flush()

        # 运行初始化
        sess.run(init)
        # Start the training loop.
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time
            # 输出
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # 更新事件文件
                summary_str = sess.run(mergeSummary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                # 写入checkpoint 以便恢复使用  有了这两句在tensorbord的embedding才有图
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)
                # 进行评估使用
                print('Training Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.train)
                print('Validation Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)
                print('Test Data Eval:')
                do_eval(sess, eval_correct, images_placeholder, labels_placeholder, data_sets.test)


# 定义main
def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)  # logs下已存在数据，则删除重建，执行训练
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


# 参数的默认初始化
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=2000,
        help='Number of steps to run trainer.'
    )
    parser.add_argument(
        '--hidden1',
        type=int,
        default=128,
        help='Number of units in hidden layer 1.'
    )
    parser.add_argument(
        '--hidden2',
        type=int,
        default=32,
        help='Number of units in hidden layer 2.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.  Must divide evenly into the dataset sizes.'
    )
    parser.add_argument(
        '--input_data_dir',
        type=str,
        default='mnist_data/',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs/fullyConnectedFeed',
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--fake_data',
        default=False,
        help='If true, uses fake data for unit testing.',
        action='store_true'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
