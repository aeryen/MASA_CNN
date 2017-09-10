import tensorflow as tf
import logging
import tensorflow.contrib.rnn as rnn


class SentenceCNN(object):
    """
    CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling layers.
    Lacks an output layer.
    """

    def __init__(
            self, prev_comp, data,
            filter_size_lists, num_filters,
            dropout=0.0, batch_normalize=False, elu=False, fc=[], l2_reg_lambda=0.0):

        self.previous_output = prev_comp.last_layer
        self.sequence_length = data.sequence_length
        self.embedding_size = data.embedding_size

        self.filter_size_lists = filter_size_lists
        self.num_filters = num_filters
        self.dropout = dropout
        self.batch_normalize = batch_normalize
        self.elu = elu
        self.l2_reg_lambda = l2_reg_lambda

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.l2_sum = tf.constant(0.0)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        num_filters_total = self.num_filters * len(self.filter_size_lists[0])

        for i, filter_size in enumerate(self.filter_size_lists[0]):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.previous_output,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # conv ==> [1, sequence_length - filter_size + 1, 1, 1]
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                # [batch_size * sentence, 1, 1, num_filters]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features

        self.h_pool = tf.concat(values=pooled_outputs, axis=3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout-keep"):
            self.last_layer = tf.nn.dropout(self.h_pool_flat, self.dropout)

    def get_last_layer_info(self):
        return self.last_layer
