# -*- coding:utf-8 -*-

import tensorflow as tf

# 读取图片原始数据
image_raw_data = tf.gfile.FastGFile('./rose.jpg').read()

with tf.Session() as sess:
    # 解码图片数据
    image_data = tf.image.decode_jpeg(image_raw_data)
    # 将图片数据类型转换为实数型
    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
    # 将图片缩小一点，这样可视化能让标注框更加清楚
    image_data = tf.image.resize_images(image_data, [180, 267], method=1)
    # tf.image.draw_bounding_boxes 函数要求图像矩阵中的数字为实数，所以
    # 需要先将图像矩阵转换为实数类型。tf.image.draw_bounding_boxes 函数
    # 的图像的输入是一个batch数据，也就是多张图像组成的四维矩阵，所以需要
    # 将解码之后的图像矩阵加一维。
    batched = tf.expand_dims(image_data, 0)

    # 给出每一张图像的所有标注框，一个标注框有4个数字，分别代表[y_min, x_min, y_max, x_max]
    # 注意这里给出的数字都是图像的相对位置。比如在180x267的图像中，
    # [0.35, 0.47, 0.5, 0.56]代表了从(63, 125) 到 (90, 150)的图像。
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])

    # 对图片进行标注
    result = tf.image.draw_bounding_boxes(batched, boxes)
    # 转换数据类型为整数型
    result = tf.image.convert_image_dtype(result, dtype=tf.uint8)
    # 降维
    result = tf.squeeze(result, 0);
    # 编码图像
    bounding = tf.image.encode_jpeg(result.eval())

    # 图像输出到文件
    with tf.gfile.GFile('./bounding_rose.jpg', 'wb') as f:
        f.write(bounding.eval())
