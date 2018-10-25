# -*- coding:utf-8 -*-

import tensorflow as tf

# 读入图片数据
image_raw_data = tf.gfile.FastGFile('./rose.jpg').read()

with tf.Session() as sess:
    # 对图像数据进行解码
    image_data = tf.image.decode_jpeg(image_raw_data)
    # 提前转换成实数类型
    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)

    # 对图像进行上下翻转
    flipped_up_down = tf.image.flip_up_down(image_data)
    # 对图像进行左右翻转
    flipped_left_right = tf.image.flip_left_right(image_data)
    # 对图像沿着对角线进行翻转
    flipped_transpose = tf.image.transpose_image(image_data)
    # 对图像进行随机上下翻转
    random_flipped_up_down = tf.image.random_flip_up_down(image)
    # 对图像进行随机左右翻转
    random_flipped_left_right = tf.image_random_flip_left_right(image)

    # 转换类型为整型
    flipped_up_down = tf.image.convert_image_dtype(flipped_up_down, dtype=tf.uint8)
    flipped_left_right = tf.image.convert_image_dtype(flipped_left_right, dtype=tf.uint8)
    flipped_transpose = tf.image.convert_image_dtype(flipped_transpose, dtype=tf.uint8)
    # 编码图像信息
    encoded_flipped_up_down = tf.image.encode_jpeg(flipped_up_down.eval())
    encoded_flipped_left_right = tf.image.encode_jpeg(flipped_left_right.eval())
    encoded_flipped_transpose = tf.image.encode_jpeg(flipped_transpose.eval())
    # 图像输出到文件
    with tf.gfile.GFile("./flip_up_down_rose.jpg", "wb") as f:
        f.write(encoded_flipped_up_down.eval())
    with tf.gfile.GFile("./flip_left_right_rose.jpg", "wb") as f:
        f.write(encoded_flipped_left_right.eval())
    with tf.gfile.GFile("./flip_transpose_rose.jpg", "wb") as f:
        f.write(encoded_flipped_transpose.eval())
