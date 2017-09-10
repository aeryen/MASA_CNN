from data_helpers.Data import DataObject

from networks.input_components.OneDocSequence import OneDocSequence
from networks.input_components.OneSequence import OneSequence

from networks.middle_components.DocumentCNN import DocumentCNN
from networks.middle_components.SentenceCNN import SentenceCNN

from networks.output_components.trip_advisor_output import TripAdvisorOutput
from networks.output_components.LSAAC2Output import LSAAC2Output


class CNNNetworkBuilder:
    """I"m currently calling this CNN builder because i'm not sure if it can handle future
    RNN parameters, and just for flexibility and ease of management the component maker is being made into
    separate function
    """
    def __init__(self, input_comp, middle_comp, output_comp):

        # input component =====
        self.input_comp = input_comp

        self.input_x = self.input_comp.input_x
        self.input_y = self.input_comp.input_y
        self.dropout_keep_prob = self.input_comp.dropout_keep_prob

        # middle component =====
        self.middle_comp = middle_comp

        # output component =====
        self.output_comp = output_comp

        self.scores = self.output_comp.scores
        self.predictions = self.output_comp.predictions
        self.loss = self.output_comp.loss
        self.accuracy = self.output_comp.accuracy

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
                             filter_size_lists=None, num_filters=None, dropout=None,
                             batch_norm=None, elu=None, fc=[], l2_reg=0.0):
        if middle_name == 'Origin':
            middle_comp = SentenceCNN(prev_comp=input_comp, data=data,
                                      filter_size_lists=filter_size_lists, num_filters=num_filters,
                                      dropout=dropout, batch_normalize=batch_norm, elu=elu,
                                      fc=fc, l2_reg_lambda=l2_reg)
        elif middle_name == "DocumentCNN":
            middle_comp = DocumentCNN(prev_comp=input_comp, data=data,
                                      filter_size_lists=filter_size_lists, num_filters=num_filters,
                                      dropout=dropout, batch_normalize=batch_norm, elu=elu,
                                      fc=fc, l2_reg_lambda=l2_reg)
        else:
            raise NotImplementedError

        return middle_comp

    @staticmethod
    def get_output_component(output_name, middle_comp, data, l2_reg=0.0):
        if "TripAdvisor" in output_name:
            output_comp = TripAdvisorOutput(prev_comp=middle_comp, data=data, l2_reg_lambda=l2_reg)
        elif "LSAAC2" in output_name:
            output_comp = LSAAC2Output(prev_comp=middle_comp, data=data, l2_reg_lambda=l2_reg)
        else:
            raise NotImplementedError

        return output_comp


if __name__ == "__main__":
    data = DataObject("test", 100)
    data.vocab = [1, 2, 3]

