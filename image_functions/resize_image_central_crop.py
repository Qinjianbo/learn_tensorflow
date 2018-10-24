# -*- coding:utf-8 -*-

import tensorflow as tf

# 读取图片原始数据
image_raw_data = tf.gfile.FastGFile('./rose.jpg').read()

with tf.Session() as sess:
    # 解码
    image_data = tf.image.decode_jpeg(image_raw_data)
    # 转换数据类型为实数类型
    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)

    # 对图片进行按比例缩放,tf.image.central_crop函数可以对图片进行按比例缩放
    # 第一个参数是原始图像，第二个参数为缩放比例，比例范围(0, 1]
    central_cropped = tf.image.central_crop(image_data, 0.3)

    # 将数据类型转化回整数型
    central_cropped = tf.image.convert_image_dtype(central_cropped, dtype=tf.uint8)

    # 对转化后的图像进行编码
    encoded_central_cropped = tf.image.encode_jpeg(central_cropped.eval())

    # 将图像输出成.jpg文件
    with tf.gfile.GFile("./central_cropped_rose.jpg", "wb") as f:
        f.write(encoded_central_cropped.eval())
