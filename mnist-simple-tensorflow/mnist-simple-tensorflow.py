# https://www.tensorflow.org/get_started/mnist/beginners

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Download the dataset and load it
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#print(mnist.train.labels[0])
#print(mnist.train.images[0])

# Input / free variables
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# The model
# matmul := matrix multiplication
y = tf.nn.softmax(tf.matmul(x, W) + b)

# The loss function
# TODO Try to use softmax_cross_entropy_with_logits here!
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Initialize and train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 100 == 0:
        #train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
        train_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        print('step %d, training accuracy %g' % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Evaluate accuracy
print('Final accuracy %g' % sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))