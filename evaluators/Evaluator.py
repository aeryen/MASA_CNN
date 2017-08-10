import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import os
import logging

from data_helpers.DataHelpers import DataHelper
import utils.ArchiveManager as AM
from data_helpers.DataHelperHotelOne import DataHelperHotelOne


class Evaluator(object):

    @staticmethod
    def get_exp_logger(exp_dir, checkpoing_file_name):
        log_path = exp_dir + checkpoing_file_name + "_eval.log"
        # logging facility, log both into file and console
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=log_path,
                            filemode='w+')
        console_logger = logging.StreamHandler()
        logging.getLogger('').addHandler(console_logger)
        logging.info("log created: " + log_path)