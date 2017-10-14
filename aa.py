from timeit import default_timer as timer
import logging
import tensorflow as tf

from utils.ArchiveManager import ArchiveManager
from networks.CNNNetworkBuilder import CNNNetworkBuilder
from trainers.TrainTask import TrainTask

from data_helpers.DataHelperHotelOne import DataHelperHotelOne

from evaluators.EvaluatorOrigin import EvaluatorOrigin
from evaluators.EvaluatorMultiAspect import EvaluatorMultiAspect
from evaluators.EvaluatorMultiAspectAAAB import EvaluatorMultiAspectAAAB

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
    # LSAAC1
    # LSAAC2
    # AAAB
    ################################################

    data_name = "TripAdvisorDoc"
    input_comp_name = "Document"
    middle_comp_name = "DocumentGRU"
    output_comp_name = "LSAAC1_MASK"

    am = ArchiveManager(data_name=data_name, input_name=input_comp_name, middle_name=middle_comp_name, output_name=output_comp_name)
    get_exp_logger(am)
    logging.warning('===================================================')
    logging.debug("Loading data...")

    if data_name == "TripAdvisor":
        dater = DataHelperHotelOne(embed_dim=300, target_sent_len=512, target_doc_len=None,
                                   aspect_id=1, doc_as_sent=True)
        ev = EvaluatorOrigin(data_helper=dater)
    elif data_name == "TripAdvisorDoc":
        dater = DataHelperHotelOne(embed_dim=300, target_doc_len=100, target_sent_len=64,
                                   aspect_id=None, doc_as_sent=False, doc_level=True)
        if output_comp_name != "AAAB":
            ev = EvaluatorMultiAspect(data_helper=dater)
        else:
            ev = EvaluatorMultiAspectAAAB(data_helper=dater)
    else:
        raise NotImplementedError

    graph = tf.Graph()
    with graph.as_default():
        input_comp = CNNNetworkBuilder.get_input_component(input_name=input_comp_name, data=dater.get_train_data())
        middle_comp = CNNNetworkBuilder.get_middle_component(middle_name=middle_comp_name, input_comp=input_comp,
                                                             data=dater.get_train_data(),
                                                             filter_size_lists=[[3, 4, 5]], num_filters=100,
                                                             batch_norm=None, elu=None,
                                                             hidden_state_dim=256, fc=[])
        output_comp = CNNNetworkBuilder.get_output_component(output_name=output_comp_name,
                                                             input_comp=input_comp,
                                                             middle_comp=middle_comp,
                                                             data=dater.get_train_data(), l2_reg=0.1, fc=[])

        tt = TrainTask(data_helper=dater, am=am,
                       input_component=input_comp,
                       middle_component=middle_comp,
                       output_component=output_comp,
                       batch_size=32, total_step=6000, evaluate_every=500, checkpoint_every=500, max_to_keep=10,
                       restore_path=None)

        start = timer()
        # n_fc variable controls how many fc layers you got at the end, n_conv does that for conv layers

        tt.training(dropout_keep_prob=1.0, batch_norm=False)
        end = timer()
        print("total training time: " + str(end - start))

        ev.evaluate(am.get_exp_dir(), None, doc_acc=True, do_is_training=True)
