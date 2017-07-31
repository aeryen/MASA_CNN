import tensorflow as tf
import numpy as np


class BetterCNN(object):
    """
    Simply use all of the features of a single doc to compute a single class distribution.
    Should be better than origin?
    """

    def __init__(
            self, num_sentence, num_word, num_aspects, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,
            init_embedding=None):
        per_aspect_sway_size = 9  # -4 0 +4
        # Placeholders for input, output and dropout, First None is batch size.
        self.input_x = tf.placeholder(tf.int32, [None, num_sentence, num_word], name="input_x")
        self.input_y_all = tf.placeholder(tf.float32, [None, num_classes], name="input_y_all")

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
            print(("self.embedded_chars " + str(self.embedded_chars.get_shape())))

            # reshape to [batch * sent_size, num_word, embed_size]
            self.flat_sentence_dim = tf.reshape(self.embedded_chars, [-1, num_word, embedding_size])
            print(("self.flat_sentence_dim " + str(self.flat_sentence_dim.get_shape())))

            # drop_sent_embed = tf.nn.dropout(self.flat_sentence_dim, self.dropout_keep_prob, name="drop_sent_embed")

            self.embedded_chars_expanded = tf.expand_dims(self.flat_sentence_dim, -1)
            print(("self.embedded_chars_expanded " + str(self.embedded_chars_expanded.get_shape())))

        # Create a convolution + maxpool layer for each filter size
        pooled_sentence_outputs = []
        num_filters_total = num_filters * len(filter_sizes)

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # l2_loss += tf.nn.l2_loss(W)
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                print(("conv of filter_size " + str(filter_size) + ": " + str(conv.get_shape())))
                pre_activation = tf.nn.bias_add(conv, b)
                h = tf.nn.relu(pre_activation, name="relu")
                # h = tf.maximum(0.01*pre_activation, pre_activation, name="relu")
                # Max-pooling over the outputs
                # [batch_size * sentence, 1, 1, num_filters]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, num_word - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print(("pooled of filter_size " + str(filter_size) + ": " + str(pooled.get_shape())))
                pooled_sentence_outputs.append(pooled)  # of sentence j of filter size f_s

        # Combine all the pooled features
        self.h_pool = tf.concat(3, pooled_sentence_outputs)
        print(("self.h_pool " + str(self.h_pool.get_shape())))
        # [batch_size * sentence, num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print(("self.h_pool_flat " + str(self.h_pool_flat.get_shape())))
        # pooled_review_outputs.append(pooled_sentence_outputs_over_f_size)

        # Add dropout
        with tf.name_scope("dropout-keep"):
            # [batch_size * sentence, num_filters_total]
            self.h_drop_sentence = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob, name="h_drop_sentence")
            print(("self.h_drop_sentence " + str(self.h_drop_sentence.get_shape())))
            self.h_drop_review = tf.reshape(self.h_drop_sentence, [-1, num_sentence * num_filters_total],
                                            name="h_drop_review")
            print(("self.h_drop_review " + str(self.h_drop_review.get_shape())))

        # per sentence rate * 6
        with tf.name_scope("rating-per-aspect"):
            with tf.name_scope("overall"):
                W = tf.get_variable(
                    name="W_all",
                    shape=[num_sentence * num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_all")
                l2_loss += tf.nn.l2_loss(W)
                # scores.shape = [batch_size * sentence, num_classes]
                self.overall_scores = tf.nn.xw_plus_b(self.h_drop_review, W, b, name="scores_all")
                self.overall_distribution = tf.nn.softmax(self.overall_scores, name="dist_all")
                print(("aspect_distribution all : " + str(self.overall_distribution.get_shape())))

        # W = tf.get_variable(
        #     "W_pred_a" + str(),
        #     shape=[num_filters_total, num_classes],
        #     initializer=tf.contrib.layers.xavier_initializer())

        # Final (unnormalized) scores and predictions
        aspect_sways = []
        with tf.name_scope("output"):
            self.overall_value = tf.argmax(self.overall_distribution, 1, name="overall_value")
            print(("self.overall_value " + str(self.overall_value.get_shape())))

        self.rate_percentage = [0, 0, 0, 0, 0]
        with tf.name_scope("overall-prediction-ratio"):
            for i in range(num_classes):
                rate1_logistic = tf.equal(self.overall_value, i)
                self.rate_percentage[i] = tf.reduce_mean(tf.cast(rate1_logistic, "float"),
                                                         name="rate-" + str(i) + "/percentage")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss-lbd" + str(l2_reg_lambda)):
            # sum_along_rating = tf.reduce_sum(batch_sent_rating_aspect_score, 1)
            # self.normalized_dist = tf.div(batch_sent_rating_aspect_score, sum_along_rating)
            losses = 0.0
            aspect_losses = \
                tf.nn.softmax_cross_entropy_with_logits(self.overall_scores,
                                                        self.input_y_all,
                                                        name="aspect_all_loss")
            losses = tf.add(losses, tf.reduce_mean(aspect_losses), name="aspect_loss_sum")
            self.loss = losses + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            with tf.name_scope("aspect_all"):
                aspect_prediction = (tf.equal(self.overall_value,
                                              tf.argmax(self.input_y_all, 1)))
                self.accuracy = tf.reduce_mean(tf.cast(tf.pack(aspect_prediction), "float"),
                                               name="accuracy-all")
