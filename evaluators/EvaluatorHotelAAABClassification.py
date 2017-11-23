#! /usr/bin/env python

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import tensorflow as tf
import os
import logging

from data_helpers.Data import DataObject
from evaluators.Evaluator import Evaluator
from data_helpers.DataHelpers import DataHelper
import utils.ArchiveManager as AM
from data_helpers.DataHelperHotelOne import DataHelperHotelOne

aspect_name = ["Other", "All", "Value", "Room", "Location", "Cleanliness", "Service"]


class EvaluatorHotelAAABClassification(Evaluator):
    def __init__(self, data_helper: DataHelper):
        self.data_helper = data_helper
        self.test_data = self.data_helper.get_test_data()
        self.eval_log = None

    def evaluate(self, experiment_dir, checkpoint_step, doc_acc=True, do_is_training=True):
        if checkpoint_step is not None:
            checkpoint_file = experiment_dir + "/checkpoints/" + "model-" + str(checkpoint_step)
        else:
            checkpoint_file = tf.train.latest_checkpoint(experiment_dir + "/checkpoints/", latest_filename=None)
        file_name = os.path.basename(checkpoint_file)
        Evaluator.get_exp_logger(exp_dir=experiment_dir, checkpoing_file_name=file_name)

        logging.info("Evaluating: " + __file__)
        logging.info("Test for prob: " + self.data_helper.problem_name)
        logging.info(checkpoint_file)
        logging.info(AM.get_time())
        logging.info("Total number of test examples: {}".format(len(self.test_data.label_instance)))

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                input_y = graph.get_operation_by_name("input_y").outputs[0]
                input_s_len = graph.get_operation_by_name("input_s_len").outputs[0]
                input_s_count = graph.get_operation_by_name("input_s_count").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                g_output_scores = graph.get_operation_by_name("output/output_scores").outputs[0]

                # Generate batches for one epoch
                x_batches = DataHelper.batch_iter(self.test_data.value, 32, 1, shuffle=False)
                y_batches = DataHelper.batch_iter(self.test_data.label_instance, 32, 1, shuffle=False)
                s_len_batches = DataHelper.batch_iter(self.test_data.sentence_len_trim, 32, 1, shuffle=False)

                # Collect the predictions here
                all_rating_score = []
                for x_test_batch, y_test_batch, s_test_batch in zip(x_batches, y_batches, s_len_batches):
                    [o_output_scores] = sess.run([g_output_scores],
                                                 {input_x: x_test_batch,
                                                  input_y: y_test_batch,
                                                  input_s_len: s_test_batch,
                                                  dropout_keep_prob: 1.0})
                    # [reviews aspects scores]
                    all_rating_score.append(o_output_scores)

                all_rating_score = np.concatenate(all_rating_score, axis=0)

                rating_pred = np.argmax(all_rating_score, axis=2)
                rating_true = np.argmax(self.test_data.label_instance, axis=2)

        with open(experiment_dir + '/aspect_rating.out', 'wb') as f:
            np.savetxt(f, rating_pred, fmt='%d', delimiter='\t')

        logging.info("Total number of OUTPUT instances: " + str(len(rating_pred)))
        for aspect_index in range(self.test_data.num_aspects):
            acc = accuracy_score(y_true=rating_true[:, aspect_index], y_pred=rating_pred[:, aspect_index])
            logging.info("ACC\ta" + str(aspect_index) + "\t" + str(acc))

            mse = mean_squared_error(y_true=rating_true[:, aspect_index], y_pred=rating_pred[:, aspect_index])
            logging.info("MSE\ta" + str(aspect_index) + "\t" + str(mse))
            # correct_predictions = float(sum(all_predictions == y_test))
            # average_accuracy = all_predictions.sum(axis=0) / float(all_predictions.shape[0])
            # print "\t" + str(average_accuracy)
            # print("Accuracy: {:g}".format(average_accuracy / float(len(y_test))))


if __name__ == "__main__":
    experiment_dir = "E:\\Research\\Paper 02\\MASA_CNN\\runs\\" \
                     "TripAdvisorDoc_Document_DocumentGRU_AAAB\\171005_1507263604\\"
    checkpoint_steps = [1000, 2000, 3000]

    dater = DataHelperHotelOne(embed_dim=300, target_doc_len=100, target_sent_len=64,
                               aspect_id=None, doc_as_sent=False, doc_level=True)

    for step in checkpoint_steps:
        # dater = DataHelperHotelOne(embed_dim=300, target_sent_len=1024, target_doc_len=None,
        #                            aspect_id=1, doc_as_sent=True)
        ev = EvaluatorHotelAAABClassification(data_helper=dater)
        ev.evaluate(experiment_dir=experiment_dir, checkpoint_step=step)