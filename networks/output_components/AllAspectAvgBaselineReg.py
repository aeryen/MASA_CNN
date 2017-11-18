import tensorflow as tf
import logging

from data_helpers.Data import DataObject


class AllAspectAvgBaselineReg(object):
    def __init__(self, input_comp, prev_comp, data: DataObject, l2_reg_lambda):
        self.label = input_comp.input_y
        self.s_count = input_comp.input_s_count
        self.prev_output = prev_comp.last_layer
        self.prev_layer_size = self.prev_output.get_shape()[-1].value

        self.document_length = data.target_doc_len
        self.num_aspects = data.num_aspects
        self.num_classes = data.num_classes

        if prev_comp.l2_sum is not None:
            self.l2_sum = prev_comp.l2_sum
            logging.warning("OPTIMIZING PROPER L2")
        else:
            self.l2_sum = tf.constant(0.0)

        self.prev_output = tf.reshape(self.prev_output, [-1, self.prev_layer_size],
                                      name="sentence_features")

        # per sentence score
        self.rating_scores = []
        with tf.name_scope("rating-score"):
            with tf.name_scope("aspect"):
                for aspect_index in range(0, self.num_aspects):
                    # W = tf.get_variable(
                    #     "W_asp"+str(aspect_index),
                    #     shape=[self.prev_layer_size, self.num_classes],
                    #     initializer=tf.contrib.layers.xavier_initializer())
                    W = tf.Variable(tf.truncated_normal([self.prev_layer_size, 1], stddev=0.1),
                                    name="W_asp" + str(aspect_index))
                    b = tf.Variable(tf.constant(0.1, shape=[1]), name="b_asp" + str(aspect_index))
                    self.l2_sum += tf.nn.l2_loss(W)
                    # [batch_size * sentence]
                    aspect_rating_score = tf.nn.xw_plus_b(self.prev_output, W, b, name="score_asp")
                    # scores = tf.reshape(scores, [-1], name="score_a" + str(aspect_index))
                    print(("aspect_rating_score " + str(aspect_rating_score.get_shape())))
                    self.rating_scores.append(aspect_rating_score)

        # Final (un-normalized) scores and predictions
        with tf.name_scope("output"):
            scaled_aspect = tf.stack(self.rating_scores, axis=1)
            print(("scaled_aspect " + str(scaled_aspect.get_shape())))

            # TODO VERY CAREFUL HERE
            batch_sent_rating_aspect = tf.reshape(scaled_aspect, [-1, self.document_length, self.num_aspects],
                                                  name="output_score")
            print(("batch_sent_rating_aspect " + str(batch_sent_rating_aspect.get_shape())))

            # [review, 6 aspect, rating]
            self.scores = tf.reduce_sum(batch_sent_rating_aspect, 1, name="output_scores")
            print(("batch_review_aspect_score " + str(self.scores.get_shape())))

            self.predictions = tf.round(self.scores, name="output_value")

        # CalculateMean Square Loss
        self.mse_aspects = []
        self.value_y = tf.cast(tf.argmax(self.label, 2, name="y_value"), dtype=tf.float32)
        with tf.name_scope("loss-lbd" + str(l2_reg_lambda)):
            self.sqr_diff_sum = tf.constant(0.0)
            aspect_square_diff = tf.squared_difference(self.scores, self.value_y,
                                                       name="review_aspect_sq_diff")
            for aspect_index in range(self.num_aspects):
                aspect_mse = tf.reduce_mean(aspect_square_diff[:, aspect_index], name="mse_a" + str(aspect_index))
                self.mse_aspects.append(aspect_mse)
                self.sqr_diff_sum = tf.add(self.sqr_diff_sum, aspect_mse, name="aspect_loss_sum")
            self.loss = self.sqr_diff_sum + l2_reg_lambda * self.l2_sum

        # Accuracy
        with tf.name_scope("accuracy"):
            self.aspect_accuracy = []
            for aspect_index in range(self.num_aspects):
                with tf.name_scope("aspect_" + str(aspect_index)):
                    aspect_prediction_logic = tf.equal(tf.round(self.scores[:, aspect_index]),
                                                       self.value_y[:, aspect_index])
                    self.aspect_accuracy.append(tf.reduce_mean(tf.cast(tf.stack(aspect_prediction_logic), "float"),
                                                               name="accuracy-" + str(aspect_index)))
            self.accuracy = tf.reduce_mean(tf.cast(tf.stack(self.aspect_accuracy), "float"), name="average-accuracy")
