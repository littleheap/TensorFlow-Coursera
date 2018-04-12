import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)  # 1.路径 2.把标签转换为01格式

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size  # 整除

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# 将输出的信号值转化为概率值
# prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义神经网络中间层（中间层是二十个神经元）
# 定义一个784*20的权值矩阵
Weights_L1 = tf.Variable(tf.random_normal([784, 20]))
# 定义一个1*20的偏值矩阵
biases_L1 = tf.Variable(tf.zeros([1, 20]))
# 定义网络总和函数
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
# 激活函数，用双曲正切函数作用于信号输出的总和
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层
# 定义一个20*10的权值矩阵
Weights_L2 = tf.Variable(tf.random_normal([20, 10]))
# 定义一个1*10的偏值矩阵
biases_L2 = tf.Variable(tf.zeros([1, 10]))
# 定义网络总和函数（输出层的输入就是中间层的输出）
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
# 激活函数，用双曲正切函数作用于信号输出的总和，即预测值
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中，生成1*100布尔矩阵
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率，现将布尔类型矩阵转换为浮点类型矩阵
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100000):

        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        if epoch % 500 == 0:
            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("Trained Times: " + str(epoch) + " , Testing Accuracy: " + str(acc))
