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

aspect_name = ["All", "Value", "Room", "Location", "Cleanliness", "Service"]

# Eval Parameters
dir_code = 1469442132


class EvaluatorMultiAspect(Evaluator):
    def __init__(self, dater):
        self.dater = dater
        self.test_data = self.dater.get_test_data()
        self.eval_log = None

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
                log_device_placement=True)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                g_output_scores = graph.get_operation_by_name("output/output_scores").outputs[0]
                g_softmax_related = graph.get_operation_by_name("related/softmax_related").outputs[0]

                # Generate batches for one epoch
                x_batches = DataHelper.batch_iter(self.test_data.value, 64, 1, shuffle=False)
                y_batches = DataHelper.batch_iter(self.test_data.label_instance, 64, 1, shuffle=False)

                # Collect the predictions here
                all_score = []
                all_dist_aspect = []
                all_max_aspect = []
                review_index = 0
                for x_test_batch, y_test_batch in zip(x_batches, y_batches):
                    # output_scores
                    # [review, 6 aspect, rating]
                    # softmax_related
                    # [batch_size * sentence, num_aspects]
                    [o_output_scores, o_softmax_related] = sess.run([g_output_scores, g_softmax_related],
                                                                    {input_x: x_test_batch, input_y: y_test_batch,
                                                                     dropout_keep_prob: 1.0})
                    # [aspect review]
                    all_score.append(o_output_scores)

                all_score = tf.concat(values=all_score, axis=0)
                batch_predictions = tf.argmax(all_score, 2)

                sent_relate_max = tf.argmax(o_softmax_related, 1).eval()
                for sent_index in range(o_output_scores.shape[1]):
                    all_dist_aspect.append(o_softmax_related[sent_index * 100:sent_index * 100 + s_count[review_index]])
                    all_max_aspect.append(sent_relate_max[sent_index * 100:sent_index * 100 + s_count[review_index]])
                    review_index += 1

                y_value = np.argmax(y_test, axis=2)

                all_score = np.concatenate(all_score, axis=0)
                all_dist_aspect = np.concatenate(all_dist_aspect, axis=0)
                with sess.as_default():
                    all_dist_aspect = tf.nn.softmax(all_dist_aspect).eval()
                all_max_aspect = np.concatenate(all_max_aspect, axis=0)
                sentence_aspect_names = [aspect_name[i] for i in all_max_aspect]

        np.savetxt("./runs/" + str(dir_code) + '/aspect_rating.out', all_score, fmt='%1.2f')
        np.savetxt("./runs/" + str(dir_code) + '/aspect_dist.out', all_dist_aspect, fmt='%1.5f')
        np.savetxt("./runs/" + str(dir_code) + '/aspect_related.out', all_max_aspect, fmt='%1.0f')

        aspect_name_file = open("./runs/" + str(dir_code) + '/aspect_related_name.out', 'w')
        for item in sentence_aspect_names:
            aspect_name_file.write("%s\n" % item)
        aspect_name_file.close()

        print(("Total number of test examples: {}".format(len(y_test))))
        for aspect_index in range(6):
            correct_predictions = all_score[:, aspect_index] == y_value[:, aspect_index]
            accuracy = np.mean(correct_predictions.astype(float))
            print("accuracy\t" + str(aspect_index) + "\t" + str(accuracy))

            mse = np.mean((all_score[:, aspect_index] - y_value[:, aspect_index]) ** 2)
            print("MSE\t" + str(aspect_index) + "\t" + str(mse))
        # correct_predictions = float(sum(all_predictions == y_test))
        # average_accuracy = all_predictions.sum(axis=0) / float(all_predictions.shape[0])
        # print "\t" + str(average_accuracy)
        # print("Accuracy: {:g}".format(average_accuracy / float(len(y_test))))
