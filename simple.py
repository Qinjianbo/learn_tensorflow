# encoding: utf8

import tensorflow as tf

# Numpy是一个科学计算的工具包，这里通过NumPy工具包生成模拟数据集
from numpy.random import RandomState
# 每批训练数据集大小
batch_size = 8

# 定义权重
w1 = tf.Variable(tf.random_normal([2, 3], stddev = 1, seed = 1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev = 1, seed = 1))

# 定义输入参数
x = tf.placeholder(tf.float32, shape = (None, 2), name = "x-input")
y_ = tf.placeholder(tf.float32, shape = (None, 1), name = "y-input")

# 前向传递过程,矩阵乘法
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 构建反向传播过程
# 定义损失函数
y = tf.sigmoid(y)
# 交叉熵计算
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    + (1-y_) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
# 生成dataset_size 大小的训练数据
X = rdm.rand(dataset_size, 2)
print "所有训练数据:"
print len(X)
print X

# 定义训练数据集标签，将x1 + x2 < 1 的认为是正样本，其他的为负样本
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]
print "所有样本标签："
print len(Y)
print Y

# 进行数据训练
with tf.Session() as sess:
    # 初始化所有全局变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 训练之前的权重
    print "训练之前的模型权重:"
    print sess.run(w1)
    print sess.run(w2)

    # 训练次数
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size
        end = min(start + batch_size, dataset_size)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        # 每隔1000次训练，输出一次交叉熵
        if i % 1000 == 0:
            print "本次训练数据："
            print len(X[start:end])
            print X[start:end]
            print "本次训练标签："
            print len(Y[start:end])
            print Y[start:end]
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" %(i, total_cross_entropy))

    # 训练结束后的权重
    print "训练之后的模型权重:"
    print sess.run(w1)
    print sess.run(w2)
