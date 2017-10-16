import tensorflow as tf
import logging

from data_helpers.Data import DataObject


class LSAAC1_MASK_Output(object):
    def __init__(self, input_comp, prev_comp, data: DataObject, l2_reg_lambda, fc=[]):
        self.label = input_comp.input_y
        self.s_count = input_comp.input_s_count
        self.dropout_keep_prob = input_comp.dropout_keep_prob
        self.prev_layer = prev_comp.last_layer
        self.sentence_feature_size = self.prev_layer.get_shape()[-1].value
        self.review_feature_size = data.target_doc_len * self.sentence_feature_size

        self.document_length = data.target_doc_len
        self.num_aspects = data.num_aspects
        self.num_classes = data.num_classes

        if prev_comp.l2_sum is not None:
            self.l2_sum = prev_comp.l2_sum
            logging.warning("OPTIMIZING PROPER L2")
        else:
            self.l2_sum = tf.constant(0.0)

        self.sent_features = tf.reshape(self.prev_layer, [-1, self.sentence_feature_size],
                                        name="sentence_features")
        self.review_features = tf.reshape(self.prev_layer, [-1, self.review_feature_size],
                                          name="review_features")
        # per sentence score
        self.rating_score = []
        with tf.name_scope("rating-score"):
            with tf.name_scope("overall"):
                self.overall_review_drop = tf.nn.dropout(self.review_features, keep_prob=self.dropout_keep_prob,
                                                         name="overall_review_drop")
                logging.warning(
                    "self.overall_review_drop = tf.nn.dropout(self.review_features, keep_prob=self.dropout_keep_prob)")
                if fc:
                    Wh_overall = tf.get_variable(
                        "overall_Wh",
                        shape=[self.review_feature_size, fc[0]],
                        initializer=tf.contrib.layers.xavier_initializer())
                    bh_overall = tf.Variable(tf.constant(0.1, shape=[fc[0]]), name="bh")
                    self.l2_sum += tf.nn.l2_loss(Wh_overall)

                    self.overall_hid_layer = tf.nn.xw_plus_b(self.overall_review_drop, Wh_overall, bh_overall,
                                                             name="overall_hid")
                    self.overall_hid_layer = tf.nn.elu(self.overall_hid_layer, name='overall_hid_elu')

                    self.overall_hidden_feature_size = fc[0]
                else:
                    self.overall_hid_layer = self.overall_review_drop
                    self.overall_hidden_feature_size = self.review_feature_size
                W_overall = tf.get_variable(
                    "W_all",
                    shape=[self.overall_hidden_feature_size, self.num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b_overall = tf.Variable(tf.constant(0.2, shape=[self.num_classes]), name="b_all")
                self.l2_sum += tf.nn.l2_loss(W_overall)
                # overall_scores.shape = [review, 5]
                self.overall_scores = tf.nn.xw_plus_b(self.overall_hid_layer, W_overall, b_overall, name="scores_all")
                print("self.overall_scores " + str(self.overall_scores.get_shape()))
                self.overall_distribution = tf.nn.softmax(self.overall_scores, name="dist_all")
                print("self.overall_distribution : " + str(self.overall_distribution.get_shape()))

            self.sent_features_for_aspect_score_drop = tf.nn.dropout(self.sent_features,
                                                                     keep_prob=self.dropout_keep_prob,
                                                                     name="sent_features_for_aspect_score_drop")
            logging.warning(
                "self.sent_features_for_aspect_score_drop = tf.nn.dropout(self.sent_features, keep_prob=self.dropout_keep_prob)")
            with tf.name_scope("aspect"):
                if fc:
                    Wh = tf.get_variable(
                        "rating_Wh",
                        shape=[self.sentence_feature_size, fc[0]],
                        initializer=tf.contrib.layers.xavier_initializer())
                    bh = tf.Variable(tf.constant(0.1, shape=[fc[0]]), name="bh")
                    self.l2_sum += tf.nn.l2_loss(Wh)

                    self.rating_layer = tf.nn.xw_plus_b(self.sent_features_for_aspect_score_drop, Wh, bh,
                                                        name="rating_hid")
                    self.rating_layer = tf.nn.elu(self.rating_layer, name='rating_hid_elu')

                    self.hidden_feature_size = fc[0]
                else:
                    self.rating_layer = self.sent_features_for_aspect_score_drop
                    self.hidden_feature_size = self.sentence_feature_size

                W = tf.get_variable(
                    "W_asp",
                    shape=[self.hidden_feature_size, self.num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                self.l2_sum += tf.nn.l2_loss(W)

                # logging.warning("INDIIIIIIIIIIIIIIII BIAS")
                # for aspect_index in range(1, self.num_aspects):
                #     b = tf.Variable(tf.constant(0.2, shape=[self.num_classes]), name="b_asp" + str(aspect_index))
                #     self.rating_score.append(
                #         tf.nn.xw_plus_b(self.rating_layer, W, b, name="score_asp" + str(aspect_index))
                #     )
                logging.warning("SINGLE BIAS")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b_asp")
                self.rating_score = tf.nn.xw_plus_b(self.rating_layer, W, b, name="score_asp")

                # logging.warning("SCORE WITH softmax")
                # self.rating_score = tf.nn.softmax(self.rating_score, name="score_asp_softmax")

        self.sent_features_for_attr_drop = tf.nn.dropout(self.sent_features,
                                                         keep_prob=self.dropout_keep_prob,
                                                         name="sent_features_for_attr_drop")
        logging.warning(
            "self.sent_features_for_attr_drop = tf.nn.dropout(self.sent_features, keep_prob=self.dropout_keep_prob)")
        with tf.name_scope("related"):
            if fc:
                Wh = tf.get_variable(
                    "related_Wh",
                    shape=[self.sentence_feature_size, fc[0]],
                    initializer=tf.contrib.layers.xavier_initializer())
                bh = tf.Variable(tf.constant(0.1, shape=[fc[0]]), name="bh")
                self.l2_sum += tf.nn.l2_loss(Wh)

                self.aspect_layer = tf.nn.xw_plus_b(self.sent_features_for_attr_drop, Wh, bh, name="relate_hid")
                self.aspect_layer = tf.nn.elu(self.aspect_layer, name='relate_hid_elu')

                self.hidden_feature_size = fc[0]
            else:
                self.aspect_layer = self.sent_features_for_attr_drop
                self.hidden_feature_size = self.sentence_feature_size

            W = tf.get_variable(
                "W_r",
                shape=[self.hidden_feature_size, self.num_aspects],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_aspects]), name="b_r")
            self.l2_sum += tf.nn.l2_loss(W)
            logging.warning("with aspect attr L2 loss: self.l2_sum += tf.nn.l2_loss(W)")

            self.attri_scores = tf.nn.xw_plus_b(self.aspect_layer, W, b, name="scores_related")
            # [batch_size * sentence, num_aspects]
            self.attri_dist = tf.nn.softmax(self.attri_scores, name="softmax_related")
            print(("self.related_distribution " + str(self.attri_dist.get_shape())))

        # Final (unnormalized) scores and predictions
        scaled_aspect = []
        with tf.name_scope("output"):
            for aspect_index in range(1, self.num_aspects):
                # [batch_size * sentence, num_classes]
                prob_aspect_sent = tf.tile(tf.expand_dims(self.attri_dist[:, aspect_index], -1),
                                           [1, self.num_classes])

                # aspect_rating = tf.multiply(self.rating_score[aspect_index-1], prob_aspect_sent,
                #                             name="aspect-" + str(aspect_index) + "-scale")
                aspect_rating = tf.multiply(self.rating_score, prob_aspect_sent,
                                            name="aspect-" + str(aspect_index) + "-scale")
                print(("scaled aspect_rating " + str(aspect_index) + " " + str(aspect_rating.get_shape())))

                scaled_aspect.append(aspect_rating)

            scaled_aspect = tf.stack(scaled_aspect, axis=1)
            print(("scaled_aspect " + str(scaled_aspect.get_shape())))

            # TODO VERY CAREFUL HERE
            batch_sent_rating_aspect = tf.reshape(scaled_aspect, [-1, self.document_length,
                                                                  self.num_aspects - 1, self.num_classes],
                                                  name="sentence_aspect_score")
            print(("batch_sent_rating_aspect " + str(batch_sent_rating_aspect.get_shape())))

            mask = tf.sequence_mask(self.s_count, maxlen=self.document_length)
            mask = tf.expand_dims(tf.expand_dims(mask, -1), -1)
            mask = tf.tile(mask, [1, 1, self.num_aspects - 1, self.num_classes], name="sequence_mask")
            batch_sent_rating_aspect = batch_sent_rating_aspect * tf.cast(mask, dtype=tf.float32)

            # [review, 6 aspect, rating]
            self.aspect_scores = tf.reduce_sum(batch_sent_rating_aspect, 1, name="review_aspect_score")
            print(("batch_review_aspect_score " + str(self.aspect_scores.get_shape())))

            self.scores = tf.concat([tf.expand_dims(self.overall_scores, axis=1), self.aspect_scores], axis=1,
                                    name="output_scores")

            # [review, 6 aspect]
            self.predictions = tf.argmax(self.scores, 2, name="output_value")
            print(("self.predictions " + str(self.predictions.get_shape())))

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss-lbd" + str(l2_reg_lambda)):
            # sum_along_rating = tf.reduce_sum(batch_sent_rating_aspect_score, 1)
            # self.normalized_dist = tf.div(batch_sent_rating_aspect_score, sum_along_rating)
            losses = tf.constant(0.0)
            for aspect_index in range(self.num_aspects):
                aspect_losses = \
                    tf.nn.softmax_cross_entropy_with_logits(logits=self.scores[:, aspect_index, :],
                                                            labels=self.label[:, aspect_index, :],
                                                            name="aspect_" + str(aspect_index) + "_loss")
                losses = tf.add(losses, tf.reduce_mean(aspect_losses), name="aspect_loss_sum")
            self.loss = losses + l2_reg_lambda * self.l2_sum

        # Accuracy
        with tf.name_scope("accuracy"):
            self.aspect_accuracy = []
            for aspect_index in range(self.num_aspects):
                with tf.name_scope("aspect_" + str(aspect_index)):
                    aspect_prediction = (tf.equal(self.predictions[:, aspect_index],
                                                  tf.argmax(self.label[:, aspect_index, :], 1)))
                    self.aspect_accuracy.append(tf.reduce_mean(tf.cast(aspect_prediction, "float"),
                                                               name="accuracy-" + str(aspect_index)))
            self.accuracy = tf.reduce_mean(tf.cast(tf.stack(self.aspect_accuracy), "float"), name="average-accuracy")
