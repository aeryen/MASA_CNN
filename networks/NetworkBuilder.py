import logging
from enum import Enum

from data_helpers.Data import DataObject

from networks.input_components.OneDocSequence import OneDocSequence
from networks.input_components.OneSequence import OneSequence

from networks.middle_components.SentenceCNN import SentenceCNN
from networks.middle_components.DocumentCNN import DocumentCNN
from networks.middle_components.DocumentGRU import DocumentGRU

from networks.output_components.OriginOutput import OriginOutput
from networks.output_components.LSAAC1Output import LSAAC1Output
from networks.output_components.LSAAC1_INDIBIAS_Output import LSAAC1_INDIBIAS_Output
from networks.output_components.LSAAC1_MASK_Output import LSAAC1_MASK_Output
from networks.output_components.LSAAC2Output import LSAAC2Output
from networks.output_components.LSAAR1Output import LSAAR1Output
from networks.output_components.LSAAR1Output_SentFCOverall import LSAAR1Output_SentFCOverall
from networks.output_components.LSAAR1Output_ShareScore import LSAAR1Output_ShareScore
from networks.output_components.LSAAR2Output_SentFCOverall import LSAAR2Output_SentFCOverall
from networks.output_components.AllAspectAvgBaseline import AllAspectAvgBaseline
from networks.output_components.AllAspectAvgBaselineReg import AllAspectAvgBaselineReg


class OutputNNType(Enum):
    OriginOutput = 0
    LSAAC1Output = 1
    LSAAC1_INDIBIAS_Output = 2
    LSAAC1_MASK_Output = 3
    LSAAC2Output = 4
    LSAAR1Output = 5
    LSAAR1Output_SentFCOverall = 6
    LSAAR1Output_ShareScore = 7
    LSAAR2Output_SentFCOverall = 8
    AAAB = 100
    AAABRegression = 101


class NetworkBuilder:
    """I"m currently calling this CNN builder because i'm not sure if it can handle future
    RNN parameters, and just for flexibility and ease of management the component maker is being made into
    separate function
    """

    def __init__(self, input_comp, middle_comp, output_comp):

        # input component =====
        self.input_comp = input_comp

        self.input_x = self.input_comp.input_x
        self.input_y = self.input_comp.input_y
        self.input_s_len = self.input_comp.input_s_len
        self.input_s_count = self.input_comp.input_s_count
        self.dropout_keep_prob = self.input_comp.dropout_keep_prob

        # middle component =====
        self.middle_comp = middle_comp

        self.is_training = self.middle_comp.is_training

        # output component =====
        self.output_comp = output_comp

        self.scores = self.output_comp.scores
        self.predictions = self.output_comp.predictions
        self.loss = self.output_comp.loss
        self.l2_sum = self.output_comp.l2_sum
        self.accuracy = self.output_comp.accuracy
        self.aspect_accuracy = self.output_comp.aspect_accuracy

        if "R1" in type(self.output_comp).__name__ \
                or "R2" in type(self.output_comp).__name__ \
                or "Reg" in type(self.output_comp).__name__:
            self.sqr_diff_sum = self.output_comp.sqr_diff_sum
            self.mse_aspects = self.output_comp.mse_aspects

    @staticmethod
    def get_input_component(input_name, data):
        # input component =====
        if input_name == "Sentence":
            input_comp = OneSequence(data=data)
        elif input_name == "Document":
            input_comp = OneDocSequence(data=data)
        else:
            raise NotImplementedError

        return input_comp

    @staticmethod
    def get_middle_component(middle_name, input_comp, data,
                             filter_size_lists=None, num_filters=None,
                             batch_norm=None, elu=None, hidden_state_dim=128, fc=[]):
        logging.info("setting: %s is %s", "filter_size_lists", filter_size_lists)
        logging.info("setting: %s is %s", "num_filters", num_filters)
        logging.info("setting: %s is %s", "batch_norm", batch_norm)
        logging.info("setting: %s is %s", "elu", elu)
        logging.info("setting: %s is %s", "HIDDEN_STATE_DIM", hidden_state_dim)
        logging.info("setting: %s is %s", "MIDDLE_FC", fc)

        if middle_name == 'Origin':
            middle_comp = SentenceCNN(prev_comp=input_comp, data=data,
                                      filter_size_lists=filter_size_lists, num_filters=num_filters,
                                      batch_normalize=batch_norm, elu=elu,
                                      fc=fc)
        elif middle_name == "DocumentCNN":
            middle_comp = DocumentCNN(prev_comp=input_comp, data=data,
                                      filter_size_lists=filter_size_lists, num_filters=num_filters,
                                      batch_normalize=batch_norm, elu=elu,
                                      fc=fc)
        elif middle_name == "DocumentGRU":
            middle_comp = DocumentGRU(prev_comp=input_comp, data=data,
                                      batch_normalize=batch_norm, elu=elu,
                                      hidden_state_dim=hidden_state_dim,
                                      fc=fc)
        else:
            raise NotImplementedError

        return middle_comp

    @staticmethod
    def get_output_component(output_type: OutputNNType, input_comp, middle_comp, data, l2_reg=0.0, fc=[]):
        logging.info("setting: %s is %s", "l2_reg", l2_reg)
        logging.info("setting: %s is %s", "OUTPUT_FC", fc)

        if output_type is OutputNNType.OriginOutput:
            output_comp = OriginOutput(input_comp=input_comp, prev_comp=middle_comp, data=data, l2_reg_lambda=l2_reg)
        elif output_type is OutputNNType.LSAAC1Output:
            output_comp = LSAAC1Output(input_comp=input_comp, prev_comp=middle_comp, data=data,
                                       l2_reg_lambda=l2_reg, fc=fc)
        elif output_type is OutputNNType.LSAAC1_INDIBIAS_Output:
            output_comp = LSAAC1_INDIBIAS_Output(input_comp=input_comp, prev_comp=middle_comp, data=data,
                                                 l2_reg_lambda=l2_reg, fc=fc)
        elif output_type is OutputNNType.LSAAC1_MASK_Output:
            output_comp = LSAAC1_MASK_Output(input_comp=input_comp, prev_comp=middle_comp, data=data,
                                             l2_reg_lambda=l2_reg, fc=fc)
        elif output_type is OutputNNType.LSAAC2Output:
            output_comp = LSAAC2Output(input_comp=input_comp, prev_comp=middle_comp, data=data,
                                       l2_reg_lambda=l2_reg, fc=fc)
        elif output_type is OutputNNType.LSAAR1Output:
            output_comp = LSAAR1Output(input_comp=input_comp, prev_comp=middle_comp, data=data,
                                       l2_reg_lambda=l2_reg, fc=fc)
        elif output_type is OutputNNType.LSAAR1Output_SentFCOverall:
            output_comp = LSAAR1Output_SentFCOverall(input_comp=input_comp, prev_comp=middle_comp, data=data,
                                                     l2_reg_lambda=l2_reg, fc=fc)
        elif output_type is OutputNNType.LSAAR1Output_ShareScore:
            output_comp = LSAAR1Output_ShareScore(input_comp=input_comp, prev_comp=middle_comp, data=data,
                                                  l2_reg_lambda=l2_reg, fc=fc)
        elif output_type is OutputNNType.LSAAR2Output_SentFCOverall:
            output_comp = LSAAR2Output_SentFCOverall(input_comp=input_comp, prev_comp=middle_comp, data=data,
                                                     l2_reg_lambda=l2_reg, fc=fc)
        elif output_type is OutputNNType.AAABRegression:
            output_comp = AllAspectAvgBaselineReg(input_comp=input_comp, prev_comp=middle_comp, data=data,
                                                  l2_reg_lambda=l2_reg)
        elif output_type is OutputNNType.AAAB:
            output_comp = AllAspectAvgBaseline(input_comp=input_comp, prev_comp=middle_comp, data=data,
                                               l2_reg_lambda=l2_reg)
        else:
            raise NotImplementedError

        return output_comp


if __name__ == "__main__":
    data = DataObject("test", 100)
    data.vocab = [1, 2, 3]
