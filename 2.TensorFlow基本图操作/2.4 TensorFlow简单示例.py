import tensorflow as tf
import numpy as np

# 线性回归
# 使用numpy生成100个随机点
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2

print('x_data = ', x_data)
print('y_data = ', y_data)

# 构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

# 二次代价函数（方差）
loss = tf.reduce_mean(tf.square(y_data - y))

# 定义一个梯度下降法来进行训练
optimizer = tf.train.GradientDescentOptimizer(0.2)

# 定义一个最小化代价函数
train = optimizer.minimize(loss)

# 初始化全局变量
init = tf.global_variables_initializer()

# 应用梯度下降法拟合k和b的值
with tf.Session() as sess:
    sess.run(init)
    for step in range(1001):
        sess.run(train)
        if step % 50 == 0:
            print(step, sess.run([k, b]))
