#! /usr/bin/env python

import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import os
import logging

from evaluators.Evaluator import Evaluator
from data_helpers.DataHelpers import DataHelper
import utils.ArchiveManager as AM
from data_helpers.DataHelperHotelOne import DataHelperHotelOne


class EvaluatorOrigin(Evaluator):
    def __init__(self, data_helper):
        self.dater = data_helper
        self.test_data = self.dater.get_test_data()

    def evaluate(self, experiment_dir, checkpoint_step, doc_acc=True, do_is_training=True):
        if checkpoint_step is not None:
            checkpoint_file = experiment_dir + "/checkpoints/" + "model-" + str(checkpoint_step)
        else:
            checkpoint_file = tf.train.latest_checkpoint(experiment_dir + "/checkpoints/", latest_filename=None)
        file_name = os.path.basename(checkpoint_file)
        Evaluator.get_exp_logger(exp_dir=experiment_dir, checkpoing_file_name=file_name)

        logging.info("Evaluating: " + __file__)
        logging.info("Test for prob: " + self.dater.problem_name)
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
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]

                # Generate batches for one epoch
                x_batches = DataHelper.batch_iter(self.test_data.value, 64, 1, shuffle=False)
                y_batches = DataHelper.batch_iter(self.test_data.label_instance, 64, 1, shuffle=False)

                # Collect the predictions here
                all_predictions = []

                for x_test_batch, y_test_batch in zip(x_batches, y_batches):
                    batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions], axis=0)

        # Print accuracy
        # np.savetxt('temp.out', all_predictions, fmt='%1.0f')
        index_label = np.argmax(self.test_data.label_instance, axis=1)
        correct_predictions = float(sum(all_predictions == index_label))
        # all_predictions = np.array(all_predictions)
        # average_accuracy = all_predictions.sum(axis=0) / float(all_predictions.shape[0])
        average_accuracy = correct_predictions / len(all_predictions)
        logging.info("ACC\t" + str(average_accuracy))

        mse = mean_squared_error(y_true=index_label, y_pred=all_predictions)
        logging.info(("MSE\t" + str(mse)))
        # print("Accuracy: {:g}".format(average_accuracy / float(len(y_test))))


if __name__ == "__main__":
    experiment_dir = "E:\\Research\\Paper 02\\MASA_CNN\\runs\\" \
                     "TripAdvisorDoc_Document_DocumentCNN_LSAAC2\\170915_1505520361"
    checkpoint_steps = [5000, 6000]

    for step in checkpoint_steps:
        EvaluatorOrigin.get_exp_logger(exp_dir=experiment_dir, checkpoing_file_name=str(step))

        dater = DataHelperHotelOne(embed_dim=300, target_doc_len=64, target_sent_len=512,
                                   aspect_id=None, doc_as_sent=False, doc_level=True)

        # dater = DataHelperHotelOne(embed_dim=300, target_sent_len=1024, target_doc_len=None,
        #                            aspect_id=1, doc_as_sent=True)
        ev = EvaluatorOrigin(data_helper=dater)

        ev.evaluate(experiment_dir=experiment_dir,
                    checkpoint_step=step)
