# -*- coding:utf-8 -*-

import tensorflow as tf

# 读入图片数据
image_raw_data = tf.gfile.FastGFile('./rose.jpg').read()

with tf.Session() as sess:
    # 解码图片数据
    image_data = tf.image.decode_jpeg(image_raw_data)
    # 将数据类型转换为实数型
    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
    # 调整图像亮度-0.5
    adjusted_brightness_black = tf.image.adjust_brightness(image_data, -0.5)
    # 调整图像亮度 0.5
    adjusted_brightness_bright = tf.image.adjust_brightness(image_data, 0.5)
    # 将图像亮度进行随机范围调整
    adjusted_brightness_random = tf.image.random_brightness(image_data, 0.5)
    # 色彩调整的API可能导致像素的实数值超出0.0 - 1.0的范围，因此在输出最终图像前需要
    # 将其值截断在0.0 - 1.0 范围区间，否则不仅图像无法正常可视化，以此为输出的神经网络
    # 的训练质量也可能受到影响
    # 如果对图像进行多项处理操作，那么这一截断过程应当在所有处理完成后进行。举例而言，
    # 假如对图像依次提高亮度和减少对比，那么第二个操作可能将第一个操作生成的部分过亮的
    # 像素拉回到不超过1.0的范围，因此在第一个操作后不应该立即截断。
    # 下面的样例假设截断操作在最终可视化图像前进行。
    adjusted_brightness_black = tf.clip_by_value(adjusted_brightness_black, 0.0, 1.0)
    adjusted_brightness_bright = tf.clip_by_value(adjusted_brightness_bright, 0.0, 1.0)
    adjusted_brightness_random = tf.clip_by_value(adjusted_brightness_random, 0.0, 1.0)
    # 将数据类型转换成整数型
    adjusted_brightness_black = tf.image.convert_image_dtype(adjusted_brightness_black, dtype=tf.uint8)
    adjusted_brightness_bright = tf.image.convert_image_dtype(adjusted_brightness_bright, dtype=tf.uint8)
    adjusted_brightness_random = tf.image.convert_image_dtype(adjusted_brightness_random, dtype=tf.uint8)
    # 编码图片数据
    adjusted_brightness_black = tf.image.encode_jpeg(adjusted_brightness_black.eval())
    adjusted_brightness_bright = tf.image.encode_jpeg(adjusted_brightness_bright.eval())
    adjusted_brightness_random = tf.image.encode_jpeg(adjusted_brightness_random.eval())

    with tf.gfile.GFile("./brightness_black_rose.jpg", "wb") as f:
        f.write(adjusted_brightness_black.eval())
    with tf.gfile.GFile("./brightness_bright_rose.jpg", "wb") as f:
        f.write(adjusted_brightness_bright.eval())
    with tf.gfile.GFile("./brightness_random_rose.jpg", "wb") as f:
        f.write(adjusted_brightness_random.eval())

    
    # 将图像对比度减少到0.5倍
    adjusted_contrast_05 = tf.image.adjust_contrast(image_data, 0.5)
    # 将图像对比度调整到5倍
    adjusted_contrast_5 = tf.image.adjust_contrast(image_data, 5)
    # 将对比度进行随机调整
    adjusted_contrast_random = tf.image.random_contrast(image_data, 1, 5)
    # 截断调整
    adjusted_contrast_05 = tf.clip_by_value(adjusted_contrast_05, 0.0, 1.0)
    adjusted_contrast_5 = tf.clip_by_value(adjusted_contrast_5, 0.0, 1.0)
    adjusted_contrast_random = tf.clip_by_value(adjusted_contrast_random, 0.0, 1.0)
    # 转换数据类型为整数型
    adjusted_contrast_05 = tf.image.convert_image_dtype(adjusted_contrast_05, dtype=tf.uint8)
    adjusted_contrast_5 = tf.image.convert_image_dtype(adjusted_contrast_5, dtype=tf.uint8)
    adjusted_contrast_random = tf.image.convert_image_dtype(adjusted_contrast_random, dtype=tf.uint8)
    # 编码
    adjusted_contrast_05 = tf.image.encode_jpeg(adjusted_contrast_05.eval())
    adjusted_contrast_5 = tf.image.encode_jpeg(adjusted_contrast_5.eval())
    adjusted_contrast_random = tf.image.encode_jpeg(adjusted_contrast_random.eval())
    with tf.gfile.GFile("./contrast_05_rose.jpg", "wb") as f:
        f.write(adjusted_contrast_05.eval())
    with tf.gfile.GFile("./contrast_5_rose.jpg", "wb") as f:
        f.write(adjusted_contrast_5.eval())
    with tf.gfile.GFile("./contrast_random_rose.jgp", "wb") as f:
        f.write(adjusted_contrast_random.eval())
