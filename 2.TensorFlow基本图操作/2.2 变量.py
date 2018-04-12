import tensorflow as tf

x = tf.Variable([1, 2])

a = tf.constant([3, 3])

sub = tf.subtract(x, a)

add = tf.add(x, sub)

# 初始化全局变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

# 创建一个值为0，名字为counter的变量
state = tf.Variable(0, name='counter')

# 新值为state加一
new_value = tf.add(state, 1)

# 赋值操作
update = tf.assign(state, new_value)

# 初始化全局变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))
