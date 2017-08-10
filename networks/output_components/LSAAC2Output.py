import tensorflow as tf
import logging


class LSAAOutput(object):
    def __init__(self, prev_layer, input_y, num_aspects, num_classes, document_length, l2_sum, l2_reg_lambda):
        self.input_y = input_y
        self.prev_layer = prev_layer
        self.prev_layer_size = prev_layer.get_shape()[-1].value
        self.num_classes = num_classes

        if l2_sum is not None:
            self.l2_sum = l2_sum
            logging.warning("OPTIMIZING PROPER L2")
        else:
            self.l2_sum = tf.constant(0.0)

        self.prev_layer = tf.reshape(self.prev_layer, [-1, self.prev_layer_size],
                                     name="sentence_features")

        # per sentence score
        self.rating_score = []
        with tf.name_scope("rating-score"):
            with tf.name_scope("aspect"):
                W = tf.get_variable(
                    "W_asp",
                    shape=[self.prev_layer_size, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_asp")
                self.l2_sum += tf.nn.l2_loss(W)
                # [batch_size * sentence]
                self.rating_score = tf.nn.xw_plus_b(self.prev_layer, W, b, name="score_asp")
                # scores = tf.reshape(scores, [-1], name="score_a" + str(aspect_index))
                print(("self.aspect_rating_score " + str(self.rating_score.get_shape())))

        with tf.name_scope("related"):
            W = tf.get_variable(
                "W_r",
                shape=[self.prev_layer_size, num_aspects + 1],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_aspects + 1]), name="b_r")
            self.l2_sum += tf.nn.l2_loss(W) * 0.5
            self.attri_scores = tf.nn.xw_plus_b(self.prev_layer, W, b, name="scores_related")
            # [batch_size * sentence, num_aspects]
            self.attri_dist = tf.nn.softmax(self.attri_scores, name="softmax_related")
            print(("self.related_distribution " + str(self.attri_dist.get_shape())))

        # Final (unnormalized) scores and predictions
        scaled_aspect = []
        with tf.name_scope("output"):
            for aspect_index in range(1, num_aspects + 1):
                # [batch_size * sentence, num_classes]
                prob_aspect_sent = tf.tile(tf.expand_dims(self.attri_dist[:, aspect_index], -1),
                                           [1, num_classes])

                aspect_rating = tf.multiply(self.rating_score, prob_aspect_sent,
                                            name="aspect-" + str(aspect_index) + "-scale")
                print(("scaled aspect_rating " + str(aspect_index) + " " + str(aspect_rating.get_shape())))

                scaled_aspect.append(aspect_rating)

            # scaled_aspect(6 aspect, ?, 5)
            scaled_aspect = tf.stack(scaled_aspect, axis=1)
            print(("scaled_aspect " + str(scaled_aspect.get_shape())))

            # TODO VERY CAREFUL HERE
            batch_sent_rating_aspect = tf.reshape(scaled_aspect, [-1, document_length, num_aspects, num_classes],
                                                  name="output_score")
            print(("batch_sent_rating_aspect " + str(batch_sent_rating_aspect.get_shape())))

            # [review, 6 aspect, rating]
            self.scores = tf.reduce_sum(batch_sent_rating_aspect, 1, name="output_scores")
            print(("batch_review_aspect_score " + str(self.scores.get_shape())))

            # [review, 6 aspect]
            self.predictions = tf.argmax(self.scores, 2, name="output_value")
            print(("self.predictions " + str(self.predictions.get_shape())))

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss-lbd" + str(l2_reg_lambda)):
            # sum_along_rating = tf.reduce_sum(batch_sent_rating_aspect_score, 1)
            # self.normalized_dist = tf.div(batch_sent_rating_aspect_score, sum_along_rating)
            losses = 0.0
            for aspect_index in range(num_aspects):
                aspect_losses = \
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.scores[:, aspect_index, :],
                                                            labels=self.input_y[:, aspect_index, :],
                                                            name="aspect_" + str(aspect_index) + "_loss")
                losses = tf.add(losses, tf.reduce_mean(aspect_losses), name="aspect_loss_sum")
            self.loss = losses + l2_reg_lambda * self.l2_sum

        # Accuracy
        with tf.name_scope("accuracy"):
            self.aspect_accuracy = []
            for aspect_index in range(num_aspects):
                with tf.name_scope("aspect_" + str(aspect_index)):
                    aspect_prediction = (tf.equal(self.predictions[:, aspect_index],
                                                  tf.argmax(self.input_y[:, aspect_index, :], 1)))
                    self.aspect_accuracy.append(tf.reduce_mean(tf.cast(aspect_prediction, "float"),
                                                               name="accuracy-" + str(aspect_index)))
            self.accuracy = tf.reduce_mean(tf.cast(tf.stack(self.aspect_accuracy), "float"), name="average-accuracy")
