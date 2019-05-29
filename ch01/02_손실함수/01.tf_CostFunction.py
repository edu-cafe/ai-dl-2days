# Linear Regression (w=2)

import tensorflow as tf

x = [1., 2., 3., 4.]
y = [2., 4., 6., 8.]
m = n_samples = len(x)

#rs_x = tf.reduce_sum(x)
#rs_y = tf.reduce_sum(y)
#sess = tf.Session()
#print(sess.run(rs_x))
#print(sess.run(rs_y))

w = tf.placeholder(tf.float32)
hypo = tf.multiply(x, w)        # H(x) = wx
cost = tf.reduce_sum(tf.pow(hypo - y, 2))/(m)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init);

for i in range(-20, 30):
    print("i:%d, w:%.1f, cost:%f" % (i, i*0.1, sess.run(cost, feed_dict={w: i*0.1})))

sess.close()



