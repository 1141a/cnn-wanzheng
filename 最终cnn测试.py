from skimage import io, transform
import tensorflow as tf
import numpy as np
import os  # os 处理文件和目录的模块
import glob  # glob 文件通配符模块

# 此程序作用于进行简单的预测，取5个图片来进行预测，如果有多数据预测，按照cnn.py中，读取数据的方式即可


path = 'F:/Python文件/分类花实例/test_photos/'
# 类别代表字典
flower_dict = {0: 'dasiy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers',
               4: 'tulips'}

w = 100
h = 100
c = 3


# 读取图片+数据处理
def read_img(path):
    # os.listdir(path) 返回path指定的文件夹包含的文件或文件夹的名字的列表
    # os.path.isdir(path)判断path是否是目录
    # b = [x+x for x in list1 if x+x<15 ]  列表生成式,循环list1，当if为真时，将x+x加入列表b
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []

    for idx, folder in enumerate(cate):
        # glob.glob(s+'*.py') 从目录通配符搜索中生成文件列表
        for im in glob.glob(folder + '/*.jfif'):
            # 输出读取的图片的名称
            print('reading the images:%s' % (im))
            # io.imread(im)读取单张RGB图片 skimage.io.imread(fname,as_grey=True)读取单张灰度图片
            # 读取的图片
            img = io.imread(im)
            # skimage.transform.resize(image, output_shape)改变图片的尺寸
            img = transform.resize(img, (w, h))
            # 将读取的图片数据加载到imgs[]列表中
            imgs.append(img)
            # 将图片的label加载到labels[]中，与上方的imgs索引对应
        # labels.append(idx)
    # 将读取的图片和labels信息，转化为numpy结构的ndarr(N维数组对象（矩阵）)数据信息
    return np.asarray(imgs, np.float32)


# 调用读取图片的函数，得到图片和labels的数据集
data = read_img(path)
with tf.compat.v1.Session() as sess:
    # 直接加载已经持久化的图
    saver =tf.compat.v1.train.import_meta_graph(
        'F:/Python文件/分类花实例/fc_model.ckpt-8.meta')
    # 所以只填到checkpoint所在的路径下即可，不需要填checkpoint
    saver.restore(sess, tf.train.latest_checkpoint('F:/Python文件/分类花实例/'))
    # sess：表示当前会话，之前保存的结果将被加载入这个会话
    # 设置每次预测的个数
    graph = tf.compat.v1.get_default_graph ()
    # 通过张量的名称获取张量
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x: data}

    logits = graph.get_tensor_by_name("logits_eval:0")  # eval功能等同于sess(run)

    classification_result = sess.run(logits, feed_dict)

    # 打印出预测矩阵
    # print(classification_result)
    # 打印出预测矩阵每一行最大值的索引
    # print(tf.argmax(classification_result, 1).eval())
    # 根据索引通过字典对应花的分
    output = []
    output = tf.argmax(classification_result, 1).eval()
    for i in range(len(output)):
        print("第", i + 1, "朵花预测:" + flower_dict[output[i]])