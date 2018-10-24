# -*- coding:utf-8 -*-

import tensorflow as tf

# 读入图片原始数据
image_raw_data = tf.gfile.FastGFile('./rose.jpg').read()

with tf.Session() as sess:
    # 对数据进行解码
    image_data = tf.image.decode_jpeg(image_raw_data)

    # 将数据类型转换为实数型
    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)

    # 通过tf.image.image_resize_image_with_crop_or_pad函数调整图像大小。这个
    # 函数的第一个参数为原始图像，后面两个参数是调整后的目标图像大小。如果
    # 原始图像的尺寸小于目标尺寸，则这个函数会自动填充0背景。如果原始图像的
    # 尺寸大于目标尺寸，则这个函数会自动截取原始图像居中部分。
    croped = tf.image.resize_image_with_crop_or_pad(image_data, 50, 50)
    pad = tf.image.resize_image_with_crop_or_pad(image_data, 1000, 1000)

    croped = tf.image.convert_image_dtype(croped, dtype=tf.uint8)
    pad = tf.image.convert_image_dtype(pad, dtype=tf.uint8)

    encoded_croped = tf.image.encode_jpeg(croped)
    encoded_pad = tf.image.encode_jpeg(pad)

    with tf.gfile.GFile("./croped_rose.jpg", "wb") as f:
        f.write(encoded_croped.eval())
    with tf.gfile.GFile("./pad_rose.jpg", "wb") as f:
        f.write(encoded_pad.eval())
    
