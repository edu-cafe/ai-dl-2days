import tensorflow as tf
import numpy as np

x_data = np.array([[2,3],[4,3],[6,4],[8,6],[10,7],[12,8],[14,9]])
y_data = np.array([0,0,0,1,1,1,1]).reshape(7,1)

x_in = tf.placeholder(tf.float64, shape=[None, 2])
y_in = tf.placeholder(tf.float64, shape=[None, 1])

a = tf.Variable(tf.random_normal([2,1], dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_normal([1], dtype=tf.float64, seed=0))

# sigmoid function
#y = 1/(1 + np.e**(a * x_data + b))
#y = tf.sigmoid(tf.matmul(x_in,a) + b)
z = tf.matmul(x_in, a) + b
y = tf.sigmoid(z)

# -(y*logh + (1-y)*log(1-h))
loss = -tf.reduce_mean(y_in * tf.log(y) + (1-y_in) * tf.log(1-y))

# LAB : tf.nn.sigmoid_cross_entropy_with_logits
# ???

learning_rate = 0.1
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicted = tf.cast(y > 0.5, dtype=tf.float64)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_in), dtype=tf.float64))

# learning
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3001):
        a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict={x_in:x_data, y_in:y_data})
        if (i+1) % 300 == 0:
            print("Epoch_%4d -> a1=%.4f, a2=%.4f, b=%.4f, loss=%.4f"
                  % (i+1, a_[0], a_[1], b_, loss_))

    print("----------------------------------------------------------")
    new_x = np.array([7, 6.]).reshape(1, 2)
    new_y = sess.run(y, feed_dict={x_in: new_x})
    print("공부한 시간 : %d, 과외 회수 : %d  --> 합격 가능성 : %6.2f %%" % (new_x[:,0], new_x[:,1], (new_y * 100)))

    sess.close()









