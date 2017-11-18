import tensorflow as tf
import numpy as np

from data_helpers.Data import DataObject

class DocumentCNNOld(object):
    """
    OVERALL Aspect is one of the aspect, also calculated using attribution method.
    Aspect Class is 5 star class distribution per sentence
    Aspect Class is then scaled using attribution, with throw away (other aspect) allowed.
    (so that's 7 aspect dist: 1 overall, 1 other, 5 normal aspect.)
    """

    def __init__(
            self, prev_comp, data: DataObject,
            filter_size_lists, num_filters,
            batch_normalize=False, elu=False, fc=[]):

        self.dropout = prev_comp.dropout_keep_prob
        self.prev_output = prev_comp.last_layer

        self.document_length = data.target_doc_len
        self.sequence_length = data.target_sent_len
        self.embedding_dim = data.init_embedding.shape[1]

        self.filter_size_lists = filter_size_lists
        self.num_filters = num_filters
        self.batch_normalize = batch_normalize
        self.elu = elu

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.l2_sum = tf.constant(0.0)

        # ? 64 64 300
        self.prev_output = tf.reshape(self.prev_output, [-1, data.target_sent_len, self.embedding_dim])
        self.prev_output = tf.expand_dims(self.prev_output, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        num_filters_total = num_filters * len(self.filter_size_lists[0])

        for i, filter_size in enumerate(self.filter_size_lists[0]):
            with tf.name_scope("conv-maxpool" + str(i)):
                # Convolution Layer
                # filter_shape = [1, filter_size, self.embedding_dim, self.num_filters]
                filter_shape = [filter_size, self.embedding_dim, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.prev_output,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # conv ==> [1, sequence_length - filter_size + 1, 1, 1]
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # h = [batch_size ?, 64, 1022, num_filters]
                # Max-pooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                # pooled = [batch_size * sent, 1, 1, num_filters]

                pooled_outputs.append(pooled)
                # [?, 64, 1, 300]

        # Combine all the pooled features
        self.last_layer = tf.concat(values=pooled_outputs, axis=3)

        # Add dropout
        with tf.name_scope("dropout-keep"):
            # [batch_size * sentence, num_filters_total]
            # self.last_layer = tf.nn.dropout(self.last_layer, self.dropout, name="h_drop_sentence")
            # [batch_size, sentence * num_filters_total]
            self.last_layer = tf.reshape(self.last_layer, [-1, self.document_length, num_filters_total],
                                         name="h_drop_review")

    def get_last_layer_info(self):
        return self.last_layer
