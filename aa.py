from timeit import default_timer as timer
import logging

from utils.ArchiveManager import ArchiveManager


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
    # * ML_One
    # * ML_2CH
    # * ML_Six
    # * ML_One_DocLevel
    # * PAN11
    # * PAN11_2CH
    #
    # Middle Components:
    #
    # * NParallelConvOnePoolNFC
    # * NConvDocConvNFC
    # * ParallelJoinedConv
    # * NCrossSizeParallelConvNFC
    # * InceptionLike
    # * PureRNN
    ################################################

    input_component = "ML_2CH"
    middle_component = "NCrossSizeParallelConvNFC"

    am = ArchiveManager(input_component, middle_component, truth_file=truth_file)
    get_exp_logger(am)
    logging.warning('===================================================')
    logging.debug("Loading data...")

    if input_component == "ML_One":
        x, y, vocabulary, vocabulary_inv, embed_matrix = data_helpers_oneAspect_glove.load_train_data(5)

        dater = DataHelperMLNormal(doc_level=LoadMethod.SENT, embed_type="glove",
                                   embed_dim=300, target_sent_len=50, target_doc_len=None, train_csv_file=truth_file,
                                   total_fold=5, t_fold_index=0)
        ev = evaler_one.Evaluator()
    else:
        raise NotImplementedError

    if middle_component == "ORIGIN_KIM":
        tt = ttl.TrainTask(data_helper=dater, am=am, input_component=input_component, exp_name=middle_component,
                           batch_size=64, evaluate_every=100, checkpoint_every=500, max_to_keep=8)
    else:
        tt = tr.TrainTask(data_helper=dater, am=am, input_component=input_component, exp_name=middle_component,
                          batch_size=64, evaluate_every=1000, checkpoint_every=2000, max_to_keep=6,
                          restore_path=None)
    start = timer()
    # n_fc variable controls how many fc layers you got at the end, n_conv does that for conv layers

    tt.training(filter_sizes=[[1, 2, 3, 4, 5]], num_filters=80, dropout_keep_prob=0.5, n_steps=15000, l2_lambda=0,
                dropout=True, batch_normalize=True, elu=True, fc=[128])
    end = timer()
    print((end - start))

    ev.load(dater)
    ev.evaluate(am.get_exp_dir(), None, doc_acc=True, do_is_training=True)
