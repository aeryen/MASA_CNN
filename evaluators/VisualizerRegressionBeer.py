#! /usr/bin/env python

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import tensorflow as tf
import os
import logging
import matplotlib.pyplot as plt
import pickle

from data_helpers.Data import DataObject
from evaluators.Evaluator import Evaluator
from data_helpers.DataHelpers import DataHelper
import utils.ArchiveManager as AM
from data_helpers.DataHelperBeer import DataHelperBeer
from tools.aspect_accuracy_human_beer import calc_aspect_f1

aspect_name = ["none", "appearance", "taste", "palate", "aroma"]


class VisualizerRegressionBeer(Evaluator):
    def __init__(self, data_helper: DataHelper, use_train_data=False):
        self.data_helper = data_helper
        self.use_train_data = use_train_data
        if not self.use_train_data:
            self.test_data = self.data_helper.get_test_data()
        else:
            self.test_data = self.data_helper.get_train_data()

        self.eval_log = None
        self.result_dir = None

    @staticmethod
    def get_exp_logger(exp_dir, checkpoing_file_name):
        if logging.root:
            del logging.root.handlers[:]

        log_path = exp_dir + checkpoing_file_name + "_vis.log"
        # logging facility, log both into file and console
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=log_path,
                            filemode='w+')
        console_logger = logging.StreamHandler()
        logging.getLogger('').addHandler(console_logger)
        logging.info("log created: " + log_path)

    def evaluate(self, experiment_dir, checkpoint_step, doc_acc=True, do_is_training=True,
                 global_mse_all=None, global_asp_f1=None):
        if checkpoint_step is not None:
            checkpoint_file = experiment_dir + "/checkpoints/" + "model-" + str(checkpoint_step)
        else:
            checkpoint_file = tf.train.latest_checkpoint(experiment_dir + "/checkpoints/", latest_filename=None)
        file_name = os.path.basename(checkpoint_file)
        if not self.use_train_data:
            self.result_dir = experiment_dir
        else:
            self.result_dir = experiment_dir + "\\train_data_output\\"
            os.makedirs(self.result_dir)

        VisualizerRegressionBeer.get_exp_logger(exp_dir=self.result_dir, checkpoing_file_name=file_name)

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

                g_score_overall_undiv = graph.get_operation_by_name("rating-score/overall/scores_all_un_div").outputs[0]
                g_score_overall_rev = graph.get_operation_by_name("rating-score/overall/scores_all").outputs[0]

                g_score_asp_sent = graph.get_operation_by_name("rating-score/aspect/score_asp").outputs[0]

                g_softmax_related = graph.get_operation_by_name("related/softmax_related").outputs[0]

                g_aspect_scaled = graph.get_operation_by_name("output/aspects-rating-scaled").outputs[0]
                g_aspect_sum = graph.get_operation_by_name("output/aspect_score_sum").outputs[0]
                g_aspect_final = graph.get_operation_by_name("output/aspect_value_beforeadd").outputs[0]

                g_output_scores = graph.get_operation_by_name("output/output_scores").outputs[0]

                # Generate batches for one epoch
                x_batches = DataHelper.batch_iter(data=self.test_data.value,
                                                  batch_size=1, num_epochs=1, shuffle=False)
                y_batches = DataHelper.batch_iter(data=self.test_data.label_instance,
                                                  batch_size=1, num_epochs=1, shuffle=False)
                s_len_batches = DataHelper.batch_iter(data=self.test_data.sentence_len_trim,
                                                      batch_size=1, num_epochs=1, shuffle=False)
                s_count_batches = DataHelper.batch_iter(data=self.test_data.doc_size_trim,
                                                        batch_size=1, num_epochs=1, shuffle=False)

                # Collect the predictions here
                instance_index = 0
                for x_test_batch, y_test_batch, s_len_batch, s_cnt_batch in \
                        zip(x_batches, y_batches, s_len_batches, s_count_batches):
                    [o_score_overall_undiv, o_score_overall_rev, o_score_asp_sent, o_softmax_related,
                     o_aspect_scaled, o_aspect_sum, o_aspect_final,
                     o_output_scores] = sess.run([g_score_overall_undiv,
                                                  g_score_overall_rev,
                                                  g_score_asp_sent,
                                                  g_softmax_related,
                                                  g_aspect_scaled,
                                                  g_aspect_sum,
                                                  g_aspect_final,
                                                  g_output_scores],
                                                 {input_x: x_test_batch,
                                                  input_y: y_test_batch,
                                                  input_s_len: s_len_batch,
                                                  input_s_count: s_cnt_batch,
                                                  dropout_keep_prob: 1.0})
                    logging.info("\n\n")

                    for i in range(self.test_data.doc_size_trim[instance_index]):
                        logging.info(str(i) + "\t" +
                                     " ".join(self.test_data.raw[
                                         self.test_data.target_doc_len * instance_index + i
                                         ]))

                    rating_true = np.argmax(y_test_batch, axis=2)
                    logging.info("TRUE LABLE: " + str(rating_true))

                    logging.info("o_score_overall_undiv: \n" + str(
                        o_score_overall_undiv[:self.test_data.doc_size_trim[instance_index] + 5].squeeze()))
                    logging.info("CLEAN o_score_overall_undiv: \n" +
                                 str(o_score_overall_undiv[:self.test_data.doc_size_trim[instance_index]]))

                    logging.info("o_score_overall_rev: \n" + str(o_score_overall_rev))

                    logging.info("o_score_asp_sent: \n" + str(
                        o_score_asp_sent[:self.test_data.doc_size_trim[instance_index] + 5]))
                    logging.info("CLEAN o_score_asp_sent: \n" +
                                 str(o_score_asp_sent[:self.test_data.doc_size_trim[instance_index]]))

                    logging.info("o_softmax_related: \n" + str(o_softmax_related))
                    clean_aspect_dist = o_softmax_related[:self.test_data.doc_size_trim[instance_index]]
                    logging.info("CLEAN o_softmax_related: \n" + str(clean_aspect_dist))
                    clean_aspect_max = np.argmax(clean_aspect_dist[:, :], axis=1)  # TODO limit aspect here
                    sentence_aspect_names = [aspect_name[i] for i in clean_aspect_max]  # TODO limit aspect here
                    logging.info("sentence_aspect_names: \n" + str(sentence_aspect_names))

                    logging.info(
                        "o_aspect_scaled: \n" + str(o_aspect_scaled[:self.test_data.doc_size_trim[instance_index] + 5]))
                    logging.info("CLEAN o_aspect_scaled: \n" +
                                 str(o_aspect_scaled[:self.test_data.doc_size_trim[instance_index]]))

                    logging.info("o_aspect_sum: \n" + str(o_aspect_sum))

                    logging.info("o_aspect_final: \n" + str(o_aspect_final))

                    logging.info("o_output_scores: \n" + str(o_output_scores))

                    instance_index += 1


if __name__ == "__main__":
    experiment_dir = "E:\\Research\\Paper 02\\MASA_CNN\\runs\\" \
                     "BeerAdvocateDoc_Document_DocumentGRU_LSAAR2Output_SentFCOverall\\171119_1511144793\\"
    # checkpoint_steps = [5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
    checkpoint_steps = [15000]

    # dater = DataHelperBeer(embed_dim=300, target_doc_len=64, target_sent_len=64,
    #                        aspect_id=None, doc_as_sent=False, doc_level=True)
    # pickle.dump(dater, open("beer6464.pickle", "wb"))
    dater = pickle.load(open("beer6464.pickle", "rb"))

    global_mse_all = []
    global_asp_f1 = []

    for step in checkpoint_steps:
        ev = VisualizerRegressionBeer(data_helper=dater, use_train_data=False)
        ev.evaluate(experiment_dir=experiment_dir, checkpoint_step=step,
                    global_mse_all=global_mse_all, global_asp_f1=global_asp_f1)

    fig, ax1 = plt.subplots()
    ax1.plot(checkpoint_steps, global_mse_all, 'b-')
    ax1.set_xlabel('steps')
    ax1.set_ylabel('MSE', color='b')
    ax1.tick_params('y', colors='b')
    plt.gca().invert_yaxis()

    ax2 = ax1.twinx()
    ax2.plot(checkpoint_steps, global_asp_f1, 'r-')
    ax2.set_ylabel('ASP', color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    plt.show()
