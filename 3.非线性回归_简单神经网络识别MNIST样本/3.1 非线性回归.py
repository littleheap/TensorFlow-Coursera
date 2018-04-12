import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy在(-0.5,0.5)之间生成等间距均匀200个随机点序列
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # 增加一个维度形成200*1矩阵
# 生成与x_data数量一样的上下浮动的噪音
noise = np.random.normal(0, 0.02, x_data.shape)
# y = x^2 + noise （大概像一个U）
y_data = np.square(x_data) + noise

print('x_data = ', x_data)
print('noise = ', noise)
print('y_data = ', y_data)

# 定义两个placeholder（N*1维）
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层（中间层是十个神经元）
# 定义一个1*10的权值矩阵，正态分布默认均值为0，标准差为1
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
# 定义一个1*10的偏值矩阵
biases_L1 = tf.Variable(tf.zeros([1, 10]))
# 定义网络总和函数，本程序中此处理论上为200*10全连接网络
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1
# 激活函数，用双曲正切函数作用于信号输出的总和
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层
# 定义一个10*1的权值矩阵
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
# 定义一个1*1的偏值矩阵
biases_L2 = tf.Variable(tf.zeros([1, 1]))
# 定义网络总和函数（输出层的输入就是中间层的输出）
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
# 激活函数，用双曲正切函数作用于信号输出的总和，即预测值
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 二次代价函数（方差）
loss = tf.reduce_mean(tf.square(y - prediction))
# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.Session() as sess:
    # 全局变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=3)  # 红色实线，宽度为3
    plt.show()
