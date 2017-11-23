#! /usr/bin/env python

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import tensorflow as tf
import os
import logging

from evaluators.Evaluator import Evaluator
from data_helpers.DataHelpers import DataHelper
import utils.ArchiveManager as AM
from data_helpers.DataHelperBeer import DataHelperBeer

aspect_name = ["none", "appearance", "taste", "palate", "aroma"]


class EvaluatorMultiAspectAAABRegressionBeer(Evaluator):
    def __init__(self, data_helper: DataHelper, use_train_data=False):
        self.data_helper = data_helper
        self.use_train_data = use_train_data
        if not self.use_train_data:
            self.test_data = self.data_helper.get_test_data()
        else:
            self.test_data = self.data_helper.get_train_data()

        self.eval_log = None
        self.result_dir = None

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

        Evaluator.get_exp_logger(exp_dir=self.result_dir, checkpoing_file_name=file_name)

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
                s_count_batches = DataHelper.batch_iter(self.test_data.doc_size_trim, 32, 1, shuffle=False)

                # Collect the predictions here
                all_rating_score = []
                for x_test_batch, y_test_batch, s_len_batch, s_cnt_batch in \
                        zip(x_batches, y_batches, s_len_batches, s_count_batches):
                    [o_output_scores] = sess.run([g_output_scores],
                                                 {input_x: x_test_batch,
                                                  input_y: y_test_batch,
                                                  input_s_len: s_len_batch,
                                                  input_s_count: s_cnt_batch,
                                                  dropout_keep_prob: 1.0})
                    # [reviews aspects scores]
                    all_rating_score.append(o_output_scores)

                all_rating_score = np.concatenate(all_rating_score, axis=0)

                rating_pred = all_rating_score
                rating_true = np.argmax(self.test_data.label_instance, axis=2)

        with open(self.result_dir + '\\' + str(checkpoint_step) + '_aspect_rating.out', 'wb') as f:
            np.savetxt(f, rating_pred, fmt='%d', delimiter='\t')

        logging.info("Total number of OUTPUT instances: " + str(len(rating_pred)))

        logging.info("ASP\t" + '\t'.join(map(str, range(self.test_data.num_aspects))))
        acc = []
        for aspect_index in range(self.test_data.num_aspects):
            acc.append(
                accuracy_score(y_true=rating_true[:, aspect_index], y_pred=np.round(rating_pred[:, aspect_index])))
        logging.info("ACC\t" + "\t".join(map(str, acc)))
        logging.info("AVG ALL\t" + str(np.mean(np.array(acc))))
        logging.info("AVG ASP\t" + str(np.mean(np.array(acc)[1:])))

        rating_true = rating_true / 2.0
        rating_pred = rating_pred / 2.0
        mse = []
        for aspect_index in range(self.test_data.num_aspects):
            mse.append(mean_squared_error(y_true=rating_true[:, aspect_index], y_pred=rating_pred[:, aspect_index]))
        logging.info("MSE\t" + "\t".join(map(str, mse)))
        logging.info("AVG ALL\t" + str(np.mean(np.array(mse))))
        logging.info("AVG ASP\t" + str(np.mean(np.array(mse)[1:])))
        if global_mse_all is not None:
            global_mse_all.append(np.mean(np.array(mse)))

        rating_true = rating_true / 4.0
        rating_pred = rating_pred / 4.0
        mse = []
        for aspect_index in range(self.test_data.num_aspects):
            mse.append(mean_squared_error(y_true=rating_true[:, aspect_index], y_pred=rating_pred[:, aspect_index]))
        logging.info("MSE [01]\t" + "\t".join(map(str, mse)))
        logging.info("AVG ALL [01]\t" + str(np.mean(np.array(mse))))
        logging.info("AVG ASP [01]\t" + str(np.mean(np.array(mse)[1:])))


if __name__ == "__main__":
    experiment_dir = "E:\\Research\\Paper 02\\MASA_CNN\\runs\\" \
                     "BeerAdvocateDoc_Document_DocumentCNN_AAABRegression\\171117_1510957388\\"
    # checkpoint_steps = [4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500]
    # checkpoint_steps = [9500, 10000, 10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000]
    # checkpoint_steps = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    # checkpoint_steps = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000]
    # checkpoint_steps = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
    checkpoint_steps = [4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000]
    # checkpoint_steps = [9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000]
    # checkpoint_steps = [11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000]

    dater = DataHelperBeer(embed_dim=300, target_doc_len=64, target_sent_len=64,
                           aspect_id=None, doc_as_sent=False, doc_level=True)

    global_mse_all = []
    global_asp_f1 = []
    for step in checkpoint_steps:
        # dater = DataHelperHotelOne(embed_dim=300, target_sent_len=1024, target_doc_len=None,
        #                            aspect_id=1, doc_as_sent=True)
        ev = EvaluatorMultiAspectAAABRegressionBeer(data_helper=dater)
        ev.evaluate(experiment_dir=experiment_dir, checkpoint_step=step)
