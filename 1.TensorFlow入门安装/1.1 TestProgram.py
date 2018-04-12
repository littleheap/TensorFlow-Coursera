import tensorflow as tf

a = tf.Variable([1, 0, 0, 1, 1])

b = tf.Variable([[1], [1], [1], [0], [0]])

c = tf.Variable([[True], [True], [True], [False], [False]])

cast_bool = tf.cast(b, dtype=tf.bool)

cast_float = tf.cast(b, dtype=tf.float32)

sess = tf.Session()

sess.run(tf.initialize_all_variables())

print(sess.run(cast_float))

print(sess.run(tf.reduce_mean(cast_float)))
