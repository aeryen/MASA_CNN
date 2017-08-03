import tensorflow as tf


class OneSequence:

    def __init__(self, document_length, sequence_length, num_aspect, num_classes,
                 word_vocab_size, embedding_size, init_embedding=None):
        # Placeholders for input, output and dropout, First None is batch size.
        self.input_x = tf.placeholder(tf.int32, [None, document_length, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_aspect, num_classes], name="input_y")
        self.input_s_count = tf.placeholder(tf.float32, [None], name="input_s_count")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Embedding layer
        with tf.variable_scope("embedding"):
            if init_embedding is None:
                W = tf.Variable(
                    tf.random_uniform([word_vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
            else:
                W = tf.Variable(init_embedding, name="W", dtype="float32")
            self.embedded = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_expanded = tf.expand_dims(self.embedded, -1)