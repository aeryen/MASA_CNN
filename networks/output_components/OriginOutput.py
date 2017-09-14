import tensorflow as tf
import logging


class OriginOutput(object):
    def __init__(self, prev_comp, data, l2_reg_lambda):
        self.prev_output = prev_comp.last_layer
        self.num_classes = data.num_classes
        self.label_instance = data.label_instance
        self.l2_reg_lambda = l2_reg_lambda

        if prev_comp.l2_sum is not None:
            self.l2_sum = prev_comp.l2_sum
            logging.warning("OPTIMIZING PROPER L2")
        else:
            self.l2_sum = tf.constant(0.0)

        # Final (unnormalized) scores and predictions
        with tf.variable_scope("output"):
            # W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            W = tf.get_variable(
                "W",
                shape=[self.prev_output.get_shape()[1].value, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")

            if self.l2_reg_lambda > 0:
                self.l2_sum += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.prev_output, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.variable_scope("loss-lbd" + str(self.l2_reg_lambda)):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.label_instance, logits=self.scores)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_sum
        # Accuracy
        with tf.variable_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.label_instance, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
