# tensorflow scan example: calculate x^2
import numpy as np
import tensorflow as tf

# sequence of elements we want to square
x = tf.placeholder(tf.int32, shape=(None,), name='x')


# thing to do to every element of the sequence
# notice how it always ignores the last output
def square(last, current):
    return current * current


# this is a "fancy for loop"
# it says: apply square to every element of x
square_op = tf.scan(
    fn=square,
    elems=x,
)

# run it!
with tf.Session() as session:
    o_val = session.run(square_op, feed_dict={x: [1, 2, 3, 4, 5]})
    print("output:", o_val)
