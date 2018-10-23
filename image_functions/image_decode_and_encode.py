# -*- coding:utf-8 -*-

# 说明：该脚本可运行
# matplotlib.pyplot 是一个python 的画图工具。后面将使用该工具来可视化经过Tensorflow
# 处理的图像
import matplotlib.pyplot as plt
import tensorflow as tf

# 读取图像的原始数据
image_raw_data = tf.gfile.FastGFile("./rose.jpg", 'r').read()

with tf.Session() as sess:
    # 对图像进行jpeg的格式解码从而得到图像对应的三维矩阵。Tensorflow
    # 还提供了tf.image.decode_png 函数对png 格式的图像进行解码。解码之后
    # 的结果为一个张量，在使用它的取值之前需要明确调用运行的过程。
    image_data = tf.image.decode_jpeg(image_raw_data)

    # 输出解码之后的三维矩阵
    print image_data.eval()

    # 利用pyplot 工具可视化得到的图像。
    plt.imshow(image_data.eval())
    plt.show()
    
    # 将表示一张图像的三维矩阵重新按照jpeg 格式编码并存入文件中。打开图像，
    # 可以得到和原始图像一样的图像
    encoded_image = tf.image.encode_jpeg(image_data)
    with tf.gfile.GFile("./rebuild_rose.jpg", 'wb') as f:
        f.write(encoded_image.eval())
