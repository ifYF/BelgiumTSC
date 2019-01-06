import tensorflow as tf
from skimage import transform
from skimage import data
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.color import rgb2gray
#import random
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

## 加载数据
def load_data(data_dir):

    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(data.imread(f))
#            a = np.zeros(62)
#            a[d] = 1
            labels.append(d)
    return images, labels

## 绝对路径加载
ROOT_PATH = "C:/Users/Administrator/Desktop/PY/BelgiumTSC" ## 当前文件夹
train_data_dir = os.path.join(ROOT_PATH, "TrafficSigns/Training")
test_data_dir = os.path.join(ROOT_PATH, "TrafficSigns/Testing")

images, labels = load_data(train_data_dir)

# Resize images修改为28*28
images32 = [transform.resize(image, (28, 28)) for image in images]
images32 = np.array(images32)
## 图像换成灰度图像
images32 = rgb2gray(np.array(images32))
##label转成float型
#labels = [float(label) for label in labels]
#for label in labels:
#    a = np.zeros(62)
#    a[label] = 1
#    label = a
images32 = np.array(images32)
labels = np_utils.to_categorical(labels)
#images32 = 
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(images32, labels, test_size = 0.1, random_state=random_seed)
class Image1:
    X_train = X_train
    X_val = X_val
    Y_train = Y_train
    Y_val = Y_val

# =============================================================================
#    _index_in_epoch = 0
#    def __init__(self,
#                 images,
#                 labels,
#                 fake_data=False,
#                 one_hot=False,
#                 dtype=dtypes.float32,
#                 reshape=True,
#                 seed=None):
##    """Construct a DataSet.
##    one_hot arg is used only if fake_data is true.  `dtype` can be either
##    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
##    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
##    """
#        seed1, seed2 = random_seed.get_seed(seed)
#        # If op level seed is not set, use whatever graph level seed is returned
#        np.random.seed(seed1 if seed is None else seed2)
#        dtype = dtypes.as_dtype(dtype).base_dtype
#        if dtype not in (dtypes.uint8, dtypes.float32):
#            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %dtype)
#        if fake_data:
#            self._num_examples = 10000
#            self.one_hot = one_hot
#        else:
#            assert images.shape[0] == labels.shape[0], (
#                    'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
#            self._num_examples = images.shape[0]
#
#            # Convert shape from [num examples, rows, columns, depth]
#            # to [num examples, rows*columns] (assuming depth == 1)
#            if reshape:
#                assert images.shape[3] == 1
#                images = images.reshape(images.shape[0],
#                                    images.shape[1] * images.shape[2])
#            if dtype == dtypes.float32:
#                # Convert from [0, 255] -> [0.0, 1.0].
#                images = images.astype(np.float32)
#                images = np.multiply(images, 1.0 / 255.0)
#        self._images = images
#        self._labels = labels
#        self._epochs_completed = 0
#        self._index_in_epoch = 0
#
#    @property
#    def images(self):
#        return self._images
#
#    @property
#    def labels(self):
#        return self._labels
#
#    @property
#    def num_examples(self):
#        return self._num_examples
#
#    @property
#    def epochs_completed(self):
#        return self._epochs_completed
#
#    def next_batch(self, batch_size, fake_data=False, shuffle=True):
##      """Return the next `batch_size` examples from this data set."""
#        if fake_data:
#            fake_image = [1] * 784
#            if self.one_hot:
#                fake_label = [1] + [0] * 9
#            else:
#                fake_label = 0
#            return [fake_image for _ in range(batch_size)], [
#                    fake_label for _ in range(batch_size)
#                    ]
#        start = self._index_in_epoch
#        # Shuffle for the first epoch
##    '''
##        第1个epoch时打乱
##    '''
#        if self._epochs_completed == 0 and start == 0 and shuffle:
#            perm0 = np.arange(self._num_examples)
#            np.random.shuffle(perm0)
#            self._images = self.images[perm0]
#            self._labels = self.labels[perm0]
#        # Go to the next epoch
##     '''
##     1个epoch最后不够1个batch还剩下几个
##     '''
#        if start + batch_size > self._num_examples:
#            # Finished epoch
#            self._epochs_completed += 1
#            # Get the rest examples in this epoch
#            rest_num_examples = self._num_examples - start
#            images_rest_part = self._images[start:self._num_examples]
#            labels_rest_part = self._labels[start:self._num_examples]
#            # Shuffle the data
##          '''打乱data'''
#            if shuffle:
#                perm = np.arange(self._num_examples)
#                np.random.shuffle(perm)
#                self._images = self.images[perm]
#                self._labels = self.labels[perm]
#            # Start next epoch
##            '''开始下一个epoch'''
#            start = 0
#            self._index_in_epoch = batch_size - rest_num_examples
#            end = self._index_in_epoch
#            images_new_part = self._images[start:end]
#            labels_new_part = self._labels[start:end]
#            return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
#        else:
#            self._index_in_epoch += batch_size
#            end = self._index_in_epoch
#            return self._images[start:end], self._labels[start:end]
# =============================================================================
image1 = Image1()

## 建立神经网络
def compute_accuracy(v_xs, v_ys): ## 定义准确度输出
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape): ##返回weights变量
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):  ##返回biases变量
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W): ## 2维卷积网络
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # SAME表示大小不变 

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    ##ksize池化窗口大小，池化核
    
batch_xs = image1.X_train
batch_ys = image1.Y_train
#batch_xs1 = tf.contrib.layers.flatten(batch_xs)
#print (batch_xs1.dtype)
#batch_xs2 = tf.cast(batch_xs1, tf.float32)
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 28,28])/255.   # 28x28
#xs = tf.placeholder(tf.float64, [None, 784])/255.   # 28x28
#xs = tf.placeholder(tf.float32, [None, 28, 28])/255.   # 28x28
ys = tf.placeholder(tf.float32, [None, 62])
#ys = tf.placeholder(tf.float32, shape = [None])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##卷积层 + 池化层
W_conv1 = weight_variable([5,5, 1,32]) # 增加图片的厚度
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                          # output size 14x14x32

## conv2 layer ##卷积层 + 池化层
W_conv2 = weight_variable([5,5, 32, 64])                 # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                          # output size 7x7x64

## fc1 layer ##全连通层
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# 全连通层62个类

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## fc2 layer ##
W_fc2 = weight_variable([1024, 62])
b_fc2 = bias_variable([62])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()

## 初始化
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
#    batch_xs, batch_ys = image1.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
#        print(compute_accuracy(
#            image1.images[:1000], image1.labels[:1000]))
        print(compute_accuracy(
            image1.X_val, image1.Y_val))
        

#fig = plt.figure(figsize=(10, 10))
#for i in range(len(sample_images)):
#    truth = sample_labels[i]
#    prediction = predicted[i]
#    plt.subplot(5, 2,1+i)
#    plt.axis('off')
#    color='green' if truth == prediction else 'red'
#    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), 
#             fontsize=12, color=color)
#    plt.imshow(sample_images[i])
#
#plt.show()







