import tensorflow as tf
import logging

from data_helpers.Data import DataObject


class LSAAR1Output(object):
    def __init__(self, input_comp, prev_comp, data: DataObject, l2_reg_lambda, fc=[]):
        self.label = input_comp.input_y
        self.s_len = input_comp.input_s_len
        self.s_count = input_comp.input_s_count
        self.dropout = input_comp.dropout_keep_prob

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
        with tf.name_scope("rating-score"):
            with tf.name_scope("overall"):
                if fc and fc[0] > 0:
                    Wh_overall = tf.get_variable(
                        "overall_Wh",
                        shape=[self.review_feature_size, fc[0]],
                        initializer=tf.contrib.layers.xavier_initializer())
                    bh_overall = tf.Variable(tf.constant(0.1, shape=[fc[0]]), name="bh")
                    self.l2_sum += tf.nn.l2_loss(Wh_overall)

                    self.overall_hid_layer = tf.nn.xw_plus_b(self.review_features, Wh_overall, bh_overall,
                                                             name="overall_hid")
                    # self.overall_hid_layer = tf.nn.sigmoid(self.overall_hid_layer, name='overall_hid_elu')

                    self.overall_hidden_feature_size = fc[0]
                else:
                    self.overall_hid_layer = self.review_features
                    self.overall_hidden_feature_size = self.review_feature_size
                W_overall = tf.get_variable(
                    "W_all",
                    shape=[self.overall_hidden_feature_size, 1],
                    initializer=tf.contrib.layers.xavier_initializer())
                b_overall = tf.Variable(tf.constant(0.1, shape=[1]), name="b_all")
                self.l2_sum += tf.nn.l2_loss(W_overall)

                # overall_scores.shape = [review, 5]
                self.overall_scores = tf.nn.xw_plus_b(self.overall_hid_layer, W_overall, b_overall,
                                                      name="scores_all_un_div")
                print("scores_all_un_div " + str(self.overall_scores.get_shape()))

                self.overall_scores = tf.div(self.overall_scores,
                                             tf.expand_dims(self.s_count, [-1]),
                                             name="scores_all")

            with tf.name_scope("aspect"):
                if fc and fc[0] > 0:
                    Wh = tf.get_variable(
                        "rating_Wh",
                        shape=[self.sentence_feature_size, fc[0]],
                        initializer=tf.contrib.layers.xavier_initializer())
                    bh = tf.Variable(tf.constant(0.1, shape=[fc[0]]), name="bh")
                    self.l2_sum += tf.nn.l2_loss(Wh)

                    self.rating_layer = tf.nn.xw_plus_b(self.sent_features, Wh, bh, name="rating_hid")
                    # self.rating_layer = tf.nn.sigmoid(self.rating_layer, name='rating_hid_elu')

                    self.hidden_feature_size = fc[0]
                else:
                    self.rating_layer = self.sent_features
                    self.hidden_feature_size = self.sentence_feature_size

                W = tf.get_variable(
                    "W_asp",
                    shape=[self.hidden_feature_size, 1],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[1]), name="b_asp")
                self.l2_sum += tf.nn.l2_loss(W)
                # [batch_size * sentence]
                self.rating_score = tf.nn.xw_plus_b(self.rating_layer, W, b, name="score_asp")
                print(("self.aspect_rating_score " + str(self.rating_score.get_shape())))
                #
                # self.rating_score = 4.0 * tf.sigmoid(self.rating_score, name="score_asp_sigmoid")

        with tf.name_scope("related"):
            # logging.info("USING CNN BUILD IN DROP NOW")
            logging.info("TRUE RELATE DROP, disabled middle layer drop")
            logging.info("sent_feat_related_drop = tf.nn.dropout(self.sent_features, self.dropout)")
            sent_feat_related_drop = tf.nn.dropout(self.sent_features, self.dropout, name="sent_feat_related_drop")

            if fc and fc[1] > 0:
                Wh = tf.get_variable(
                    "related_Wh",
                    shape=[self.sentence_feature_size, fc[1]],
                    initializer=tf.contrib.layers.xavier_initializer())
                bh = tf.Variable(tf.constant(0.1, shape=[fc[1]]), name="bh")
                self.l2_sum += 0.2 * tf.nn.l2_loss(Wh)

                self.aspect_layer = tf.nn.xw_plus_b(sent_feat_related_drop, Wh, bh, name="relate_hid")
                self.aspect_layer = tf.nn.elu(self.aspect_layer, name='relate_hid_elu')

                self.hidden_feature_size = fc[1]
            else:
                self.aspect_layer = sent_feat_related_drop
                self.hidden_feature_size = self.sentence_feature_size

            W = tf.get_variable(
                "W_r",
                shape=[self.hidden_feature_size, self.num_aspects],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_aspects]), name="b_r")
            self.l2_sum += 0.2 * tf.nn.l2_loss(W)

            self.attri_scores = tf.nn.xw_plus_b(self.aspect_layer, W, b, name="scores_related")
            # [batch_size * sentence, num_aspects]
            self.attri_dist = tf.nn.softmax(self.attri_scores, name="softmax_related")
            print(("self.related_distribution " + str(self.attri_dist.get_shape())))

        # Final (unnormalized) scores and predictions
        scaled_aspect = []
        with tf.name_scope("output"):
            aspect_rating_score = tf.tile(self.rating_score, [1, self.num_aspects - 1])
            print(("self.aspect_rating_score " + str(aspect_rating_score.get_shape())))

            prob_aspect_sent = self.attri_dist[:, 1:]
            print(("prob_aspect_sent " + str(prob_aspect_sent.get_shape())))

            aspects_rating_scaled = tf.multiply(aspect_rating_score, prob_aspect_sent, name="aspects-rating-scaled")
            print(("scaled aspect_rating " + str(aspects_rating_scaled.get_shape())))

            # TODO VERY CAREFUL HERE
            # batch_sent_rating_aspect shape = [batch, sentence, 5 aspects]
            batch_sent_aspect_rating = tf.reshape(aspects_rating_scaled,
                                                  [-1, self.document_length, self.num_aspects - 1],
                                                  name="batch_sent_aspect_rating")
            print(("batch_sent_aspect_rating " + str(batch_sent_aspect_rating.get_shape())))

            # [review, 5 aspect]
            self.batch_review_aspect_score = tf.reduce_sum(batch_sent_aspect_rating, axis=1, name="aspect_score_sum")

            # [batch_size, sentence, num_aspects]
            batch_sent_aspect_prob = tf.reshape(self.attri_dist, [-1, self.document_length, self.num_aspects])
            print(("batch_sent_aspect_prob " + str(batch_sent_aspect_prob.get_shape())))

            # [review 64?, 5 aspect]
            review_aspect_prob_sum = tf.reduce_sum(batch_sent_aspect_prob, 1, name="relate_ratio_sum")[:, 1:]

            self.aspect_final_value = tf.div(self.batch_review_aspect_score, review_aspect_prob_sum,
                                             name="aspect_value_beforeadd")

            self.scores = tf.concat(values=[self.overall_scores, self.aspect_final_value],
                                    axis=1,
                                    name="output_scores")
            # self.review_final_value = tf.maximum(tf.minimum(review_final_value, 4.0), 0.0, name="review_final_value")
            print(("review_final_value " + str(self.scores.get_shape())))

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
