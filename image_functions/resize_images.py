# -*- coding:utf-8 -*-

import tensorflow as tf

# 读取原始图像数据
image_data_raw = tf.gfile.FastGFile('./rose.jpg').read()

with tf.Session() as sess:
    # 对图像进行解码
    image_data = tf.image.decode_jpeg(image_data_raw)

    print image_data.eval()

    # 首先将图片数据转化为实数类型。这一步将0~255的像素值转化为0.0~1.0范围内的实数
    # 大多数图像处理API支持证书和实数类型输入。如果输入是整数类型，这些API会
    # 在内部将输入转化为实数后处理，再将输出转化为整数。如果有多个处理步骤，在整数和
    # 实数之间的反复转化将导致精度损失，因此推荐在图像处理前将其转化为实数类型。
    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)

    print image_data.eval()

    resized = tf.image.resize_images(image_data, [300, 300], method = 0)

    print resized.eval()

    resized = tf.image.convert_image_dtype(resized, dtype=tf.uint8)

    encoded_image = tf.image.encode_jpeg(resized)
    with tf.gfile.GFile("./resize_rose_300_300.jpg", "wb") as f:
        f.write(encoded_image.eval())
