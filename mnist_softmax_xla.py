# coding=utf-8
"""
一个带有JIT XLA的一个简单例子
"""
import argparse
import sys

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import timeline

FLAGS = None


# main里面实现inference  loss  train  evaluation
def main(_):
    # 导入数据
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # inference
    x = tf.placeholder(tf.float32, [None, 784])
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, w) + b

    y_ = tf.placeholder(tf.float32, [None, 10])

    # loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # 配置session启动参数
    config = tf.ConfigProto()
    jit_level = 0
    if FLAGS.xla:
        # 打开xla jit优化器
        jit_level = tf.OptimizerOptions.ON_1

    config.graph_options.optimizer_options.global_jit_level = jit_level
    run_metadata = tf.RunMetadata()

    sess = tf.Session(config=config)
    tf.global_variables_initializer().run(session=sess)

    # Train
    train_loops = 1000
    for i in range(train_loops):
        batch_xs, batch_ys = mnist.train.next_batch(100)

        """
    timeline用于可视化tensorflow各节点执行时间线，从op粒度展示tensorflow执行时间和并发情况
    为最后一次loop循环创建timeline，使用json保存的，可以使用下面的地址查看
    chrome://tracing/
    """
        if i == train_loops - 1:
            sess.run(train_step,
                     feed_dict={x: batch_xs,
                                y_: batch_ys},
                     # RunOptions里面定义追踪级别
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                     # 定义运行元数据，这样就可以看运行的时间和内存情况
                     run_metadata=run_metadata)
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            with open('timeline.ctf.json', 'w') as trace_file:
                trace_file.write(trace.generate_chrome_trace_format())
        else:
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 预测
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    sess.close()


if __name__ == '__main__':
    """
  1.import argparse
  2.调用ArgumentParser()实例化一个对象
  3.add_argument()添加参数
  4.使用parse_known_args()进行解析，成功解析的放在FLAGS里，解析失败的放在unparse里
  """
    parser = argparse.ArgumentParser()  # 命令行参数解析
    parser.add_argument(
        '--data_dir',  # 参数name
        type=str,  # 参数的类型
        default='mnist_data',  # 默认值
        help='Directory for storing input data')  # 帮组信息，当错误使用时候提示的信息
    """
  xla保证tensorflow的灵活性，有的简单操作会影响性能包括由简单操作组合的操作，使用
  xla对这些操作进行一个优化，提高性能。JIT--just in time及时变编译
  """
    parser.add_argument(
        '--xla', type=bool, default=True, help='Turn xla via JIT on')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs/fullyConnectedFeed',
        help='Directory to put the log data.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    """
  app.run 实现参数的解析和main()方法的执行
  """
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
