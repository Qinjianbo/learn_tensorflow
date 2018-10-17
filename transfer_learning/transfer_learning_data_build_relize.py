# -*- coding: utf-8 -*-

import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 说明：该文件可运行，但是内存占用太大，导致本机无法执行
# 数据下载地址
# wget http://download.tensorflow.org/example_images/flower_photos.tgz
# tar xzf flower_photos.tgz

# 原始输入数据的目录，这个目录下有5个子目录，每个子目录下面保存属于该类别的所有图片
# /path/to/flower_photos
INPUT_DATA = '/home/bob/Documents/learn_tensorflow/flower_photos'

# 输出文件地址。将整理后的图片数据通过numpy的格式保存。
# /path/to/flower_processed_data.npy
OUTPUT_FILE = '/home/bob/Documents/learn_tensorflow/python_script/flower_processed_data.npy'

# 测试数据和验证数据比例
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10

# 读取数据并将数据分割成训练数据、验证数据和测试数据
def create_image_lists(sess, testing_percentage, validation_percentage) :
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    print 'sub_dirs', sub_dirs
    is_root_dir = True

    # 初始化各个数据集
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0 

    isPrint = True
    # 读取所有的子目录
    for sub_dir in sub_dirs:
        print 'deal_sub_dir:', sub_dir
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取一个子目录中所有的图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue
        print 'file_list_count', len(file_list)
        print 'file_list[0:5]', file_list[0:5]

        # 处理图片数据
        for file_name in file_list:
            # 读取并解析图片，将图片转换成299x299以便incption-v3模型来处理
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()
            image = tf.image.decode_jpeg(image_raw_data)
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(
                    image, dtype = tf.float32)
            image = tf.image.resize_images(image, [299, 299])
            image_value = sess.run(image)
            # 打印一次结果，供分析
            if isPrint:
                print 'image_value:', image_value
                print 'image_value_count:', len(image_value)
                print 'image_value[0]_count:', len(image_value[0])
                print 'image_value[0][0]_count:', len(image_value[0][0])
                isPrint = False

            # 随机分数据集
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (testing_percentage + validation_percentage) :
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
        current_label += 1

    # 将训练数据随机打乱以获得更好的训练效果
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)
    
    return np.asarray([training_images, training_labels,
                    validation_images, validation_labels,
                    testing_images, testing_labels])

# 数据整理主函数
def main():
    with tf.Session() as sess:
        print 'create_image_lists_begin'
        processed_data = create_image_lists(
            sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        print 'create_image_lists_end'
        # 通过numpy 格式保存处理后的数据
        np.save(OUTPUT_FILE, processed_data)

if __name__ == '__main__':
    main()
