from timeit import default_timer as timer
import logging

from utils.ArchiveManager import ArchiveManager

from data_helpers.DataHelperHotelOne import DataHelperHotelOne
from evaluators.EvaluatorOrigin import EvaluatorOrigin
from trainers.TrainTask import TrainTask

def get_exp_logger(am):
    log_path = am.get_exp_log_path()
    # logging facility, log both into file and console
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_path,
                        filemode='w+')
    console_logger = logging.StreamHandler()
    logging.getLogger('').addHandler(console_logger)
    logging.info("log created: " + log_path)


if __name__ == "__main__":

    ###############################################
    # exp_names you can choose from at this point:
    #
    # Input Components:
    #
    # * TripAdvisor
    #
    # Middle Components:
    #
    # * Origin
    #
    #
    #
    ################################################

    input_component = "TripAdvisorDoc"
    middle_component = "DocumentCNN"
    output_component = "LSAA"

    am = ArchiveManager(input_component, middle_component)
    get_exp_logger(am)
    logging.warning('===================================================')
    logging.debug("Loading data...")

    if input_component == "TripAdvisor":
        dater = DataHelperHotelOne(embed_dim=300, target_sent_len=512, target_doc_len=None,
                                   aspect_id=1, doc_as_sent=True)
        ev = EvaluatorOrigin(dater=dater)
    elif input_component == "TripAdvisorDoc":
        dater = DataHelperHotelOne(embed_dim=300, target_doc_len=64, target_sent_len=512,
                                   aspect_id=None, doc_as_sent=False, doc_level=True)
        ev = EvaluatorOrigin(dater=dater)
    else:
        raise NotImplementedError

    tt = TrainTask(data_helper=dater, am=am,
                   input_component=input_component, middle_component=middle_component,
                   output_component=output_component,
                   batch_size=32, evaluate_every=500, checkpoint_every=5000, max_to_keep=6,
                   restore_path=None)

    start = timer()
    # n_fc variable controls how many fc layers you got at the end, n_conv does that for conv layers

    tt.training(filter_sizes=[[3, 4, 5]], num_filters=100, l2_lambda=0, dropout_keep_prob=0.75,
                batch_normalize=False, elu=False, fc=[], n_steps=5000)
    end = timer()
    print((end - start))

    ev.evaluate(am.get_exp_dir(), None, doc_acc=True, do_is_training=True)
