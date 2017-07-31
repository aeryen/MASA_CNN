import tensorflow as tf
import numpy as np


class RegressCNN(object):
    """
    Overall Score is direct fully connected layer for 1 score. Normalized using document sentence count.
    Aspect Score is 1 score per sentence (not 6 per sentence)
    Aspect Score is then scaled using attribution, with throw away (other aspect) allowed.
    Final Document Aspect Score is a deviation from the Overall Score! (Added together)
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
            print(("self.embedded_chars " + str(self.embedded_chars.get_shape())))

            # reshape to [batch * sent_size, num_word, embed_size]
            self.flat_sentence_dim = tf.reshape(self.embedded_chars, [-1, num_word, embedding_size])
            print(("self.flat_sentence_dim " + str(self.flat_sentence_dim.get_shape())))

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
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                print(("conv of filter_size " + str(filter_size) + ": " + str(conv.get_shape())))
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
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

        # Combine all the pooled features along dim 3
        self.h_pool = tf.concat(3, pooled_sentence_outputs)
        print(("self.h_pool " + str(self.h_pool.get_shape())))
        # [batch_size * sentence, num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        print(("self.h_pool_flat " + str(self.h_pool_flat.get_shape())))
        # pooled_review_outputs.append(pooled_sentence_outputs_over_f_size)

        # Add dropout
        with tf.name_scope("dropout-keep" + str(0.5)):
            # [batch_size * sentence, num_filters_total]
            self.h_drop_sentence = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob, name="h_drop_sentence")
            print(("self.h_drop_sentence " + str(self.h_drop_sentence.get_shape())))
            # [batch_size, sentence * num_filters_total]
            self.h_drop_review = tf.reshape(self.h_drop_sentence, [-1, num_sentence * num_filters_total],
                                            name="h_drop_review")
            print(("self.h_drop_review " + str(self.h_drop_review.get_shape())))

        self.aspect_rating_score = []
        with tf.name_scope("rating-score"):
            """Overall Score"""
            with tf.name_scope("overall"):
                W = tf.get_variable(
                    "W_all",
                    shape=[num_sentence * num_filters_total, 1],
                    initializer=tf.contrib.layers.xavier_initializer())
                # W = tf.Variable(tf.truncated_normal([num_sentence * num_filters_total, 1], stddev=0.1), name="W_all")
                b = tf.Variable(tf.constant(0.1, shape=[1]), name="b_all")
                l2_loss += tf.nn.l2_loss(W)
                # overall_scores.shape = [review]
                overall_score_undiv = tf.nn.xw_plus_b(self.h_drop_review, W, b)
                print(("overall_score_undiv " + str(overall_score_undiv.get_shape())))
                print(("self.input_s_count " + str(self.input_s_count.get_shape())))
                """Normalization"""
                self.overall_scores = tf.div(
                    overall_score_undiv,
                    tf.expand_dims(self.input_s_count, [-1]),
                    name="scores_all")
                print(("self.overall_scores " + str(self.overall_scores.get_shape())))

            """Just 1 Aspect Score"""
            with tf.name_scope("aspect"):
                W = tf.get_variable(
                    "W_asp",
                    shape=[num_filters_total, 1],
                    initializer=tf.contrib.layers.xavier_initializer())
                # W = tf.Variable(tf.truncated_normal([num_filters_total, 1], stddev=0.1), name="W_asp")
                b = tf.Variable(tf.constant(0.1, shape=[1]), name="b_asp")
                l2_loss += tf.nn.l2_loss(W)
                # scores.shape = [batch_size * sentence] # 1 score for each sentence
                self.aspect_rating_score = tf.nn.xw_plus_b(self.h_drop_sentence, W, b, name="score_asp")
                # scores = tf.reshape(scores, [-1], name="score_a" + str(aspect_index))
                print(("self.aspect_rating_score " + str(self.aspect_rating_score.get_shape())))
        # [batch_size * sentence, num_classes] * number_aspect

        """6 aspect Aspect Attribution"""
        with tf.name_scope("related"):
            W = tf.get_variable(
                "W_r",
                shape=[num_filters_total, num_aspects],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_aspects]), name="b_r")
            l2_loss += tf.nn.l2_loss(W)
            scores = tf.nn.xw_plus_b(self.h_drop_sentence, W, b, name="scores_related")
            # [batch_size * sentence, num_aspects]
            self.related_distribution = tf.nn.softmax(scores, name="softmax_related")
            print(("self.related_distribution " + str(self.related_distribution.get_shape())))

        # Final (unnormalized) scores and predictions
        scaled_aspect = []
        with tf.name_scope("output"):
            self.aspect_rating_score = tf.tile(self.aspect_rating_score, [1, num_aspects - 1])
            print(("self.aspect_rating_score " + str(self.aspect_rating_score.get_shape())))

            # WITH THROW AWAY [1:]
            # [batch_size * sentence, num_aspects]
            prob_aspect_sent = self.related_distribution[:, 1:]
            print(("prob_aspect_sent " + str(prob_aspect_sent.get_shape())))

            aspects_rating_scaled = tf.mul(self.aspect_rating_score, prob_aspect_sent,
                                           name="aspects-rating-scaled")
            print(("scaled aspect_rating " + str(aspects_rating_scaled.get_shape())))
            # aspects_rating_scaled shape = [batch_size * sentence, ] x 6
            # (6400, 5)

            # TODO VERY CAREFUL HERE
            # batch_sent_rating_aspect shape = [batch, sentence, 5 aspects]
            batch_sent_aspect_rating = tf.reshape(aspects_rating_scaled, [-1, num_sentence, num_aspects - 1],
                                                  name="batch_sent_aspect_rating")
            print(("batch_sent_aspect_rating " + str(batch_sent_aspect_rating.get_shape())))

            # [review, 5 aspect]
            # s_count_aspect_tile = tf.tile(tf.expand_dims(self.input_s_count, [-1]), [1, 5])
            self.batch_review_aspect_score = tf.reduce_sum(batch_sent_aspect_rating, 1, name="aspect_score_sum")

            # self.aspect_final_value = tf.div(self.batch_review_aspect_score, s_count_aspect_tile,
            #                                  name="aspect_final_value")

            # [batch_size, sentence, num_aspects]
            batch_sent_aspect_prob = tf.reshape(self.related_distribution, [-1, num_sentence, num_aspects])
            print(("batch_sent_aspect_prob " + str(batch_sent_aspect_prob.get_shape())))
            # [review 64?, 5 aspect]
            review_aspect_prob_sum = tf.reduce_sum(batch_sent_aspect_prob, 1, name="relate_ratio_sum")[:, 1:]

            self.aspect_final_value = tf.div(self.batch_review_aspect_score, review_aspect_prob_sum,
                                             name="aspect_value_beforeadd")

            # [batch_size 64, 1]
            # add overall value to aspects
            # TODO =====================================================================================================
            aspect_final_value = tf.add(
                self.aspect_final_value,
                tf.tile(self.overall_scores, [1, num_aspects - 1]),
                name="aspect_final_value"
            )

            self.review_final_value = tf.concat(concat_dim=1, values=[self.overall_scores, aspect_final_value],
                                                name="review_final_value")
            # self.review_final_value = tf.maximum(tf.minimum(review_final_value, 4.0), 0.0, name="review_final_value")
            print(("review_final_value " + str(self.review_final_value.get_shape())))

        self.rate_percentage = [0, 0, 0, 0, 0]
        # with tf.name_scope("prediction-ratio"):
        #     for i in range(num_classes):
        #         rate1_logistic = tf.equal(tf.round(self.review_final_value[:, 0]), i)
        #         self.rate_percentage[i] = tf.reduce_mean(tf.cast(rate1_logistic, "float"),
        #                                                  name="rate-" + str(i) + "/percentage")

        # Calculate Square Loss
        self.mse_aspects = []
        self.value_y = tf.cast(tf.argmax(self.input_y, 2, name="y_value"), "float")
        print(("value_y " + str(self.value_y.get_shape())))
        with tf.name_scope("loss-lbd" + str(l2_reg_lambda)):
            sqr_diff_sum = 0.0
            # [64, 6]
            aspect_square_diff = tf.squared_difference(self.review_final_value, self.value_y,
                                                       name="review_aspect_sq_diff")
            print(("aspect_square_diff " + str(aspect_square_diff.get_shape())))

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
