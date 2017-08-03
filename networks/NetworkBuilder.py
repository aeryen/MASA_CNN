from networks.input_components.OneSequence import OneSequence
from networks.middle_components.OriginCNN import OriginCNN
from networks.output_components.trip_advisor_output import TripAdvisorOutput


class NetworkBuilder:
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Works for both single label (PAN) and multilabel (ML) datasets
    """

    def __init__(
            self, data, document_length, sequence_length, num_classes, embedding_size, filter_size_lists, num_filters,
            input_component, middle_component,
            l2_reg_lambda, dropout, batch_normalize, elu, fc):

        word_vocab_size = len(data.vocab)

        # input component
        if input_component.endswith("TripAdvisor"):
            self.input_comp = OneSequence(sequence_length, num_classes, word_vocab_size, embedding_size,
                                          data.embed_matrix)
        else:
            raise NotImplementedError

        self.input_x = self.input_comp.input_x
        self.input_y = self.input_comp.input_y
        self.dropout_keep_prob = self.input_comp.dropout_keep_prob

        # middle component
        if middle_component == 'Origin':
            self.middle_comp = OriginCNN(previous_component=self.input_comp,
                                         sequence_length=sequence_length, embedding_size=embedding_size,
                                         filter_size_lists=filter_size_lists, num_filters=num_filters,
                                         dropout=dropout, batch_normalize=batch_normalize, elu=elu,
                                         fc=fc, l2_reg_lambda=l2_reg_lambda)
        else:
            raise NotImplementedError

        try:
            self.is_training = self.middle_comp.is_training
        except NameError:
            # print("is_training is not defined")
            self.is_training = None
            pass

        prev_layer = self.middle_comp.get_last_layer_info()
        l2_sum = self.middle_comp.l2_sum

        # output component
        if "TripAdvisor" in data.name:
            output = TripAdvisorOutput(self.input_comp.input_y, prev_layer, num_classes, l2_sum, l2_reg_lambda)
        else:
            raise NotImplementedError

        self.scores = output.scores
        self.predictions = output.predictions

        self.loss = output.loss

        self.accuracy = output.accuracy
