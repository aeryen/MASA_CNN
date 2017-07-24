import numpy as np
import re
import itertools
import pickle

from data_helpers.DataHelpers import DataHelper
from data_helpers.Data import DataObject


class DataHelperHotelOne(DataHelper):

    def __init(self, embed_dim, target_doc_len, target_sent_len, aspect_id):
        super(DataHelperHotelOne, self).__init__(embed_dim=embed_dim, target_doc_len=target_doc_len,
                                                 target_sent_len=target_sent_len)

        self.aspect_id = aspect_id

        self.load_train_data()

    def load_data_and_labels(self):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.

        aspect_id should be 0 1 2 3 4 5
        """

        # Load data from files
        identity = [[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]]

        train_count = list(open("./data/BalanceByAspect0_LengthFix1_3000perRate_Review/aspect_0.count", "r").readlines())
        train_count = [int(s) for s in train_count if (len(s) > 0 and s != "\n")]

        train_rating = list(open("./data/BalanceByAspect0_LengthFix1_3000perRate_Review/aspect_0.rating", "r").readlines())
        train_rating = [s for s in train_rating if (len(s) > 0 and s != "\n")]
        y = [identity[int(s.split(" ")[self.aspect_id]) - 1] for s in train_rating]

        train_content = list(open("./data/BalanceByAspect0_LengthFix1_3000perRate_Review/aspect_0.txt", "r").readlines())
        train_content = [s.strip() for s in train_content]
        # Split by words
        x_text = [self.clean_str(sent) for sent in train_content]

        # review_lens = []
        x = []
        for train_line_index in range(len(x_text)):
            tokens = x_text[train_line_index].split()

            if len(tokens) > 5120:
                # print str(len(tokens)) + "\t" + x_text[train_line_index]
                x_text[train_line_index] = x_text[train_line_index].replace("\?", "")
                x_text[train_line_index] = x_text[train_line_index].replace("(", "")
                x_text[train_line_index] = x_text[train_line_index].replace(")", "")
                x_text[train_line_index] = re.sub(r'[0-9]', '', x_text[train_line_index])

                tokens = x_text[train_line_index].split()

                if len(tokens) > 6700:
                    tokens = tokens[:6700]
                    # print "\t### Force Cut"
                    # print "\t" + str(len(tokens)) + "\t" + x_text[train_line_index]
            x.append(tokens)

        data = DataObject("NAME", len(y))
        data.raw = x
        data.label_doc = y

        return data

    def load_test_data_and_labels(self, aspect_id):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        # Load data from files
        identity = [[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]]

        train_count = list(
            open("./data/BalanceByAspect0_LengthFix1_3000perRate_Review/test_aspect_0.count", "r").readlines())
        train_count = [int(s) for s in train_count if (len(s) > 0 and s != "\n")]

        train_rating = list(
            open("./data/BalanceByAspect0_LengthFix1_3000perRate_Review/test_aspect_0.rating", "r").readlines())
        train_rating = [s for s in train_rating if (len(s) > 0 and s != "\n")]
        y = [identity[int(s.split(" ")[aspect_id]) - 1] for s in train_rating]

        train_content = list(
            open("./data/BalanceByAspect0_LengthFix1_3000perRate_Review/test_aspect_0.txt", "r").readlines())
        train_content = [s.strip() for s in train_content]
        # Split by words
        x_text = [self.clean_str(sent) for sent in train_content]

        # review_lens = []
        x = []
        for train_line_index in range(len(x_text)):
            tokens = x_text[train_line_index].split()

            if len(tokens) > 5120:
                # print str(len(tokens)) + "\t" + x_text[train_line_index]
                x_text[train_line_index] = x_text[train_line_index].replace("\?", "")
                x_text[train_line_index] = x_text[train_line_index].replace("(", "")
                x_text[train_line_index] = x_text[train_line_index].replace(")", "")
                x_text[train_line_index] = re.sub(r'[0-9]', '', x_text[train_line_index])

                tokens = x_text[train_line_index].split()

                if len(tokens) > 6700:
                    tokens = tokens[:6700]
                    # print "\t### Force Cut"
                    # print "\t" + str(len(tokens)) + "\t" + x_text[train_line_index]
            x.append(tokens)

        return [x, y]

    def load_train_data(self):
        """
        Loads and preprocessed data for the MR dataset.
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        train_data = self.load_data_and_labels()
        self.vocab, self.vocab_inv = self.build_vocab(train_data)
        self.embed_matrix = self.build_glove_embedding(self.vocab_inv)
        train_data = self.build_content_vector(train_data)
        train_data = self.pad_sentences(train_data)

        self.train_data = train_data
        self.train_data.embed_matrix = self.embed_matrix
        self.train_data.vocab = self.vocab
        self.train_data.vocab_inv = self.vocab_inv

    def load_test_data(aspect_id):
        """
        Loads and preprocessed data for the MR dataset.
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        reviews, labels = load_test_data_and_labels(aspect_id=aspect_id)
        reviews_padded, labels = pad_sentences(reviews=reviews, y=labels, target_length=2312)
        # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
        vocabulary, vocabulary_inv = pickle.load(open("vocabulary.pickle", "rb"))
        x, y = build_input_data(reviews_padded, labels, vocabulary)
        return [x, y, vocabulary, vocabulary_inv]


if __name__ == "__main__":
    a = DataHelperHotelOne()
