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

    
    
    # 可通过提供标注的方式来告诉随机截图的算法哪部分是“有信息量”的
    # min_object_convered = 0.4 表示截取部分至少包含某个标注框40%的内容
    begin,size,bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image_data), bounding_boxes=boxes, min_object_covered=0.4)

    # 通过标注框可视化随机取得的图像。
    # 把图像数据升维
    batched = tf.expand_dims(image_data, 0)
    image_with_box = tf.image.draw_bounding_boxes(batched, boxes, bbox_for_draw)
    # 编码图像
    encoded_bounding = tf.image.encode_jpeg(image_with_box.eval())
    with tf.gfile.GFile('./bounding_with_box_rose.jpg', 'wb') as f:
        f.write(encoded_bounding.eval())
    # 截取随机出来的图像。因为算法是随机的，所以每次取得的结果会是不一样的
    distorted_image = tf.slice(image_data, begin, size)
    encoded_distorted_image = tf.image.encode_jpeg(distored_image.eval())
    with tf.gfile.GFile('./distorted_rose.jpg', 'wb') as f:
        f.write(encoded_distorted_image.eval())
