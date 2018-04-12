import tensorflow as tf

# 定义一个1*2矩阵常量
m1 = tf.constant([[3, 3]])

# 定义一个2*1矩阵常量
m2 = tf.constant([[2], [3]])

# 创建一个矩阵乘法操作，把m1 m2传入
product = tf.matmul(m1, m2)

# 打印输出
print(product)

# 定义一个会话，启动默认图
sess = tf.Session()

# 调用run方法执行op中的乘法操作
result = sess.run(product)

# 打印输出结果
print(result)

# 关闭会话
sess.close()

# 第二种定义会话的操作
with tf.Session() as sess:
    result = sess.run(product)
    print(result)
