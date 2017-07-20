import tensorflow as tf
import numpy as np


class RegressCNN(object):
    """
    All sentence to all aspect score
    """

    def __init__(
            self, num_sentence, num_word, num_aspects, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,
            init_embedding=None):
        # Placeholders for input, output and dropout, First None is batch size.
        self.input_x = tf.placeholder(tf.int32, [None, num_sentence, num_word], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_aspects, num_classes], name="input_y")
        self.input_s_count = tf.placeholder(tf.float32, [None], name="input_s_count")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if init_embedding is None:
                W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
            else:
                W = tf.Variable(init_embedding, name="W", dtype="float32")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x, name="embedded_chars")
            print("self.embedded_chars " + str(self.embedded_chars.get_shape()))

            # reshape to [batch * sent_size, num_word, embed_size]
            self.flat_sentence_dim = tf.reshape(self.embedded_chars, [-1, num_word, embedding_size])
            print("self.flat_sentence_dim " + str(self.flat_sentence_dim.get_shape()))

            self.embedded_chars_expanded = tf.expand_dims(self.flat_sentence_dim, -1)
            print("self.embedded_chars_expanded " + str(self.embedded_chars_expanded.get_shape()))

        # Create a convolution + maxpool layer for each filter size
        pooled_sentence_outputs = []
        num_filters_total = num_filters * len(filter_sizes)

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                print("conv of filter_size " + str(filter_size) + ": " + str(conv.get_shape()))
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Max-pooling over the outputs
                # [batch_size * sentence, 1, 1, num_filters]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, num_word - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print("pooled of filter_size " + str(filter_size) + ": " + str(pooled.get_shape()))
                pooled_sentence_outputs.append(pooled)  # of sentence j of filter size f_s

        # Combine all the pooled features along dim 3
        self.h_pool = tf.concat(3, pooled_sentence_outputs)
        print("self.h_pool " + str(self.h_pool.get_shape()))
        # [batch_size * sentence, num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print("self.h_pool_flat " + str(self.h_pool_flat.get_shape()))
        # pooled_review_outputs.append(pooled_sentence_outputs_over_f_size)

        # Add dropout
        with tf.name_scope("dropout-keep" + str(0.5)):
            # [batch_size * sentence, num_filters_total]
            self.h_drop_sentence = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob, name="h_drop_sentence")
            print("self.h_drop_sentence " + str(self.h_drop_sentence.get_shape()))
            # [batch_size, sentence * num_filters_total]
            self.h_drop_review = tf.reshape(self.h_drop_sentence, [-1, num_sentence * num_filters_total],
                                            name="h_drop_review")
            print("self.h_drop_review " + str(self.h_drop_review.get_shape()))

        self.aspect_rating_score = []
        with tf.name_scope("rating-score"):
            """Overall Score"""
            with tf.name_scope("overall"):
                W = tf.get_variable(
                    "W_all",
                    shape=[num_sentence * num_filters_total, num_aspects],
                    initializer=tf.contrib.layers.xavier_initializer())
                # W = tf.Variable(tf.truncated_normal([num_sentence * num_filters_total, 1], stddev=0.1), name="W_all")
                b = tf.Variable(tf.constant(0.1, shape=[num_aspects]), name="b_all")
                l2_loss += tf.nn.l2_loss(W)
                # overall_scores.shape = [review]
                overall_score_undiv = tf.nn.xw_plus_b(self.h_drop_review, W, b)
                print("overall_score_undiv " + str(overall_score_undiv.get_shape()))
                print("self.input_s_count " + str(self.input_s_count.get_shape()))

        # Final (unnormalized) scores and predictions
        scaled_aspect = []
        with tf.name_scope("output"):
            """Normalization"""
            self.review_final_value = tf.div(
                overall_score_undiv,
                tf.expand_dims(self.input_s_count, [-1]),
                name="review_final_value")
            print("review_final_value " + str(self.review_final_value.get_shape()))

        # self.rate_percentage = [0, 0, 0, 0, 0]
        # with tf.name_scope("prediction-ratio"):
        #     for i in range(num_classes):
        #         rate1_logistic = tf.equal(tf.round(self.review_final_value[:, 0]), i)
        #         self.rate_percentage[i] = tf.reduce_mean(tf.cast(rate1_logistic, "float"),
        #                                                  name="rate-" + str(i) + "/percentage")

        # Calculate Square Loss
        self.mse_aspects = []
        self.value_y = tf.cast(tf.argmax(self.input_y, 2, name="y_value"), "float")
        print("value_y " + str(self.value_y.get_shape()))
        with tf.name_scope("loss-lbd" + str(l2_reg_lambda)):
            sqr_diff_sum = 0.0
            # [64, 6]
            aspect_square_diff = tf.squared_difference(self.review_final_value, self.value_y,
                                                       name="review_aspect_sq_diff")
            print("aspect_square_diff " + str(aspect_square_diff.get_shape()))

            for aspect_index in range(num_aspects):
                aspect_mse = tf.reduce_mean(aspect_square_diff[:, aspect_index], name="mse_a" + str(aspect_index))
                self.mse_aspects.append(aspect_mse)
                sqr_diff_sum = tf.add(sqr_diff_sum, aspect_mse, name="aspect_loss_sum")
            self.loss = sqr_diff_sum + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            self.aspect_accuracy = []
            for aspect_index in range(num_aspects):
                with tf.name_scope("aspect_" + str(aspect_index)):
                    aspect_prediction_logic = tf.equal(tf.round(self.review_final_value[:, aspect_index]),
                                                       self.value_y[:, aspect_index])
                    self.aspect_accuracy.append(tf.reduce_mean(tf.cast(tf.pack(aspect_prediction_logic), "float"),
                                                               name="accuracy-" + str(aspect_index)))
            self.accuracy = tf.reduce_mean(tf.cast(tf.pack(self.aspect_accuracy), "float"), name="average-accuracy")
