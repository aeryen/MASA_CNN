from networks.input_components.OneSequence import OneSequence
from networks.input_components.OneDocSequence import OneDocSequence
from networks.middle_components.SentenceCNN import SentenceCNN
from networks.middle_components.DocumentCNN import DocumentCNN
from networks.output_components.trip_advisor_output import TripAdvisorOutput
from networks.output_components.LSAAOutput import LSAAOutput


class NetworkBuilder:
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    Works for both single label (PAN) and multilabel (ML) datasets
    """

    def __init__(
            self, data, document_length, sequence_length, num_aspects, num_classes,
            embedding_size, filter_size_lists, num_filters,
            input_component, middle_component, output_component,
            l2_reg_lambda, dropout, batch_normalize, elu, fc):

        vocab_size = len(data.vocab)

        # input component =====
        if input_component == "TripAdvisor":
            self.input_comp = OneSequence(sequence_length, num_classes, vocab_size, embedding_size,
                                          data.embed_matrix)
        elif input_component == "TripAdvisorDoc":
            self.input_comp = OneDocSequence(document_length=document_length, sequence_length=sequence_length,
                                             num_aspects=num_aspects, num_classes=num_classes,
                                             vocab_size=vocab_size, embedding_size=embedding_size,
                                             init_embedding=data.embed_matrix)
        else:
            raise NotImplementedError

        self.input_x = self.input_comp.input_x
        self.input_y = self.input_comp.input_y
        self.dropout_keep_prob = self.input_comp.dropout_keep_prob

        # middle component =====
        if middle_component == 'Origin':
            self.middle_comp = SentenceCNN(previous_component=self.input_comp,
                                           sequence_length=sequence_length, embedding_size=embedding_size,
                                           filter_size_lists=filter_size_lists, num_filters=num_filters,
                                           dropout=dropout, batch_normalize=batch_normalize, elu=elu,
                                           fc=fc, l2_reg_lambda=l2_reg_lambda)
        elif middle_component == "SimpleCNN":
            self.middle_comp = DocumentCNN(previous_component=self.input_comp,
                                           document_length=document_length, sequence_length=sequence_length,
                                           embedding_size=embedding_size,
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

        # output component =====
        if "TripAdvisor" in output_component:
            output = TripAdvisorOutput(self.input_comp.input_y, prev_layer, num_classes, l2_sum, l2_reg_lambda)
        elif "LSAA" in output_component:
            output = LSAAOutput(prev_layer=prev_layer, input_y=self.input_comp.input_y,
                                num_aspects=num_aspects, num_classes=num_classes,
                                document_length=document_length,
                                l2_sum=l2_sum, l2_reg_lambda=l2_reg_lambda)
        else:
            raise NotImplementedError

        self.scores = output.scores
        self.predictions = output.predictions
        self.loss = output.loss
        self.accuracy = output.accuracy
