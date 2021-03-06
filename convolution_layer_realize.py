# encoding:utf-8

import tensorflow as tf

# 说明：该脚本无法运行，input未定义

# 通过tf.get_variable 的方式创建过滤器的权重变量和偏置项变量。上面介绍了卷积层
# 的参数个数只和过滤器的尺寸、深度以及当前层节点矩阵的深度有关，所以这里声明的参数变
# 量是一个四维矩阵，前面两个维度代表了过滤器的尺寸，第三个维度代表当前层深度，第四个
# 维度标识过滤器的深度。
filter_weight = tf.get_variable(
    'weights', [5, 5, 3, 16],
    initializer=tf.truncated_normal_initializer(stddev=0.1))

# 和卷积层的权重类似，当前层矩阵上不同位置的偏置项也是共享的，所以总共有下一层深度个不
# 同的偏置项。本样例代码中16为过滤器的深度，也是神经网络中下一层节点矩阵的深度。
biases = tf.get_variable(
    'biases', [16], initializer=tf.constant_initializer(0.1))

# tf.nn.conv2d提供了一个非常方便的函数来实现卷积层前向传播算法。这个函数的第一个输入为
# 当前层的节点矩阵。注意这个矩阵是一个四维矩阵，后面三个维度对应一个节点矩阵，第一维
# 对应一个输入batch。比如输入层，input[0, :, :, :]标识第一张图片，input[1, :, :, :]
# 表示第二张图片，以此类推。tf.nn.conv2d 第二个参数提供了卷积层的权重，第三个参数为不同
# 维度上的步长。虽然第三个参数提供的是一个长度为4的数组，但是第一维和最后一维的数字要求
# 一定是1.这是因为卷积层步长只对矩阵的长和宽有效。最后一个参数是填充（padding）
# 的方法，Tensorflow 中提供SAME 或是 VALID 两种选择。其中SAME 标识添加全0填充，VALID表示
# 不添加填充。
conv = tf.nn.conv2d(
    input, filter_weight, strides=[1, 1, 1, 1], padding='SAME')

# tf.nn.bias_add提供了一个方便的函数给每一个节点加上偏置项。注意这里不能直接使用假发，
# 因为矩阵上不同位置上的节点都需要加上同样的偏置项。虽然下一层神经网络的大小为2 x 2，但是
# 偏置项只是一个数（因为深度为1），而2 x 2矩阵中的每一个值都需要加上这个偏置项。
bias = tf.nn.bias_add(conv, biases)
# 将计算结果通过ReLU激活函数完成去线性化。
actived_conv = tf.nn.relu(bias)
