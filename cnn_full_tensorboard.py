# -*- coding: utf-8 -*-

"""
@author : xzy
@time ： 2017.5
本代码包含了tensorboard 中embedding、scalar、graph等标签的所有代码
代码来自 https://github.com/tensorflow/tensorflow/issues/6322

关于tensorboard embedding的一个简介参看网站
https://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/
或者 http://blog.csdn.net/aliceyangxi1987/article/details/71079387

tensorboard 的embedding分为下面三个步骤：
1) 创建一个2D张量保存embedding数据
    embedding_var = tf.Variable(....)
2) 分时段的在LOG_DIR目录的checkpoint文件中保存模型变量
    saver = tf.train.Saver()
    saver.save(session, os.path.join(LOG_DIR, "model.ckpt"), step)
3) (可选) 将 embedding 和元数据关联
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time

from six.moves import xrange
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

import numpy as np
import scipy.misc

FLAGS = None


def placeholder_inputs(batch_size):
    """产生变量占位操作
  Args:
    batch_size: 批次大小

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """

    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, mnist.IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32, shape=batch_size)
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    """构建数据字典.

  一个feed_dict操作的形式:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: 返回一个具有映射关系的数据字典
  """
    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set,
            return_results=False):
    """验证操作

  Args:
    sess: 用于执行操作的session
    eval_correct: 包含正确预测数目的tensor
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: 验证输入的数据集
    return_results: True if the results should be returned for the embedding.

  Returns:
    all_images: A list of batches of images.
    all_labels: A list of batches of labels.
    all_hidden1_outputs: A list of batches of embeddings from the first hidden layer.
    all_hidden2_outputs: A list of batches of embeddings from the second hidden layer.
  """
    true_count = 0  # 累计正确预测数目变量
    steps_per_epoch = data_set.num_examples // FLAGS.batch_size  # 轮寻次数
    num_examples = steps_per_epoch * FLAGS.batch_size
    if return_results:
        all_images = []
        all_labels = []
        all_hidden1_outputs = []
        all_hidden2_outputs = []
        # 在RELU之前，返回给定名称的tensor，张量的命名node:num”的形式给出,
        # node是节点名,num表示当前张量来自节点的第几个输出
        hidden1_outputs = tf.get_default_graph().get_tensor_by_name('hidden1/add:0')
        hidden2_outputs = tf.get_default_graph().get_tensor_by_name('hidden2/add:0')
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set, images_placeholder, labels_placeholder)
        if return_results:
            # 向列表的尾部添加一个新的元素
            all_images.append(feed_dict[images_placeholder])
            all_labels.append(feed_dict[labels_placeholder])
            curr_count, hidden1_output, hidden2_output = sess.run(
                [eval_correct, hidden1_outputs, hidden2_outputs],
                feed_dict=feed_dict)
            true_count += curr_count
            all_hidden1_outputs.append(hidden1_output)
            all_hidden2_outputs.append(hidden2_output)
        else:
            true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))
    if return_results:
        return all_images, all_labels, all_hidden1_outputs, all_hidden2_outputs


def images_to_sprite(data):
    """从数组生成图片
    Args:
      data: 输入图片张量 Bache_size*Height*Weight[*3]
    Returns:
      data: 重构造带有padding的 Height*Weight*3 image
    """
    if len(data.shape) == 3:
        # np.tile复制各个维度，(1, 1, 1, 3)代表将最内层的张量复制3遍，比如
        # data = [[[[2, 3, 4]]]],经过data = np.tile(data, (1,1,1,3))之后
        # data = [[[[2 3 4 2 3 4 2 3 4]]]]
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min_ = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min_).transpose(3, 0, 1, 2)
    max_ = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max_).transpose(3, 0, 1, 2)
    # 反转颜色，显示更好看
    data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=0)
    # 将各个缩略图平铺成图像
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


def run_training():
    """训练模型"""
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

        # inference
        logits = mnist.inference(images_placeholder, FLAGS.hidden1, FLAGS.hidden2)
        # loss
        loss = mnist.loss(logits, labels_placeholder)
        # train
        train_op = mnist.training(loss, FLAGS.learning_rate)
        # evaluation
        eval_correct = mnist.evaluation(logits, labels_placeholder)
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        # saver
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)
        sess.run(init)

        for step in xrange(FLAGS.max_steps):
            start_time = time.time()

            # 获取 feed dict
            feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time

            # 写入 summaries 并且 打印step、loss等信息.
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # 不断更新写入每步长收集到的信息
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # 设置恢复点
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)

                # 执行训练数据集
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.train)
                # 验证测试数据集
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)
                # 验证测试集

                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.test)

        # 获取图片的单边像素，即28
        thumbnail_size = int(np.sqrt(mnist.IMAGE_PIXELS))
        for data_set, name in [
            (data_sets.train, 'train'),
            (data_sets.validation, 'validation'),
            (data_sets.test, 'test')]:
            output_path = os.path.join(FLAGS.log_dir, 'embed', name)
            print('Computing %s Embedding' % name)
            (all_images, all_labels, hidden1_vectors, hidden2_vectors) = do_eval(
                sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_set,
                True)  # 循环收集train、validation、test的embedding信息
            embed_tensors = []
            summary_writer = tf.summary.FileWriter(output_path, sess.graph)
            """建立 embedding projector"""
            config = projector.ProjectorConfig()
            # enumerate 可以同时取出 index 和 value[index]
            for layer, embed_vectors in enumerate([hidden1_vectors, hidden2_vectors]):
                embed_tensor = tf.Variable(np.array(embed_vectors).reshape(
                    len(embed_vectors) * embed_vectors[0].shape[0], -1),
                    name=('%s_layer_%s' % (name, layer)))
                embed_tensors.append(embed_tensor)
                sess.run(embed_tensor.initializer)
                # 指定想要可视化的 variable，metadata 文件的位置
                embedding = config.embeddings.add()
                embedding.tensor_name = embed_tensor.name
                embedding.metadata_path = os.path.join(output_path, 'labels.tsv')
                embedding.sprite.image_path = os.path.join(output_path, 'sprite.png')
                embedding.sprite.single_image_dim.extend([thumbnail_size, thumbnail_size])
                projector.visualize_embeddings(summary_writer, config)  # 存储一个config文件
            result = sess.run(embed_tensors)
            saver = tf.train.Saver(embed_tensors)
            saver.save(sess, os.path.join(output_path, 'model.ckpt'), layer)

            # 保存 metadata,可视化时看到不同数字用不同颜色表示，需要知道每个 image 的name-class
            images = np.array(all_images).reshape(-1, thumbnail_size,
                                                  thumbnail_size).astype(np.float32)
            sprite = images_to_sprite(images)
            scipy.misc.imsave(os.path.join(output_path, 'sprite.png'), sprite)
            all_labels = np.array(all_labels).flatten()
            metadata_file = open(os.path.join(output_path, 'labels.tsv'), 'w')
            metadata_file.write('Name\tClass\n')
            for ll in xrange(len(all_labels)):
                metadata_file.write('%06d\t%d\n' % (ll, all_labels[ll]))
            metadata_file.close()


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    run_training()


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
        default='/home/xzy/input_data',
        help='Directory to put the input data.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs/cnn-fully-tensorboard',
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
