import numpy as np
import re
import itertools
import pickle
import logging
import pkg_resources

from data_helpers.DataHelpers import DataHelper
from data_helpers.Data import DataObject


class DataHelperHotelOne(DataHelper):

    problem_name = "TripAdvisor"

    sent_num_file = ["aspect_0.count", "test_aspect_0.count"]
    rating_file = ["aspect_0.rating", "test_aspect_0.rating"]
    content_file = ["aspect_0.txt", "test_aspect_0.txt"]

    def __init__(self, embed_dim, target_doc_len, target_sent_len, aspect_id, doc_as_sent=False):
        super(DataHelperHotelOne, self).__init__(embed_dim=embed_dim, target_doc_len=target_doc_len,
                                                 target_sent_len=target_sent_len)

        logging.info("setting: %s is %s", "aspect_id", aspect_id)
        self.aspect_id = aspect_id
        logging.info("setting: %s is %s", "doc_as_sent", doc_as_sent)
        self.doc_as_sent = doc_as_sent

        self.dataset_dir = self.data_path + 'hotel_balance_LengthFix1_3000per/'
        self.num_of_classes = 5

        self.load_all_data()

    def load_files(self, load_test):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.

        aspect_id should be 0 1 2 3 4 5
        """

        sent_count = list(open(self.dataset_dir + self.sent_num_file[load_test], "r").readlines())
        sent_count = [int(s) for s in sent_count if (len(s) > 0 and s != "\n")]

        aspect_rating = list(open(self.dataset_dir + self.rating_file[load_test], "r").readlines())
        aspect_rating = [s for s in aspect_rating if (len(s) > 0 and s != "\n")]
        y = [s.split(" ")[self.aspect_id] for s in aspect_rating]
        y = np.array(list(map(float, y))) - 1
        y_onehot = self.to_onehot(y, 5)

        train_content = list(open(self.dataset_dir + self.content_file[load_test], "r").readlines())
        train_content = [s.strip() for s in train_content]
        # Split by words
        x_text = [self.clean_str(sent) for sent in train_content]

        if self.doc_as_sent:
            x_text = DataHelperHotelOne.concat_to_doc(sent_list=x_text, sent_count=sent_count)

        # review_lens = []
        x = []
        for train_line_index in range(len(x_text)):
            tokens = x_text[train_line_index].split()
            x.append(tokens)

        data = DataObject(self.problem_name, len(y))
        data.raw = x
        data.label_doc = y_onehot
        data.doc_size = sent_count

        return data

    def load_all_data(self):
        train_data = self.load_files(0)
        self.vocab, self.vocab_inv = self.build_vocab([train_data], self.vocabulary_size)
        self.embed_matrix = self.build_glove_embedding(self.vocab_inv)
        train_data = self.build_content_vector(train_data)
        train_data = self.pad_sentences(train_data)

        self.train_data = train_data
        self.train_data.embed_matrix = self.embed_matrix
        self.train_data.vocab = self.vocab
        self.train_data.vocab_inv = self.vocab_inv
        self.train_data.label_instance = self.train_data.label_doc

        test_data = self.load_files(1)
        test_data = self.build_content_vector(test_data)
        test_data = self.pad_sentences(test_data)

        self.test_data = test_data
        self.test_data.embed_matrix = self.embed_matrix
        self.test_data.vocab = self.vocab
        self.test_data.vocab_inv = self.vocab_inv
        self.test_data.label_instance = self.test_data.label_doc

if __name__ == "__main__":
    a = DataHelperHotelOne(embed_dim=300, target_doc_len=200, target_sent_len=1024, aspect_id=1, doc_as_sent=True)
