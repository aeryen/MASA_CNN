import numpy as np
import re
import logging


from data_helpers.DataHelpers import DataHelper
from data_helpers.Data import DataObject

sent_length_target = 65
sent_length_hard = 67


class DataHelperHotelAll(DataHelper):

    problem_name = "TripAdvisor"

    sent_num_file = ["aspect_0.count", "test_aspect_0.count"]
    rating_file = ["aspect_0.rating", "test_aspect_0.rating"]
    content_file = ["aspect_0.txt", "test_aspect_0.txt"]

    def __init__(self, embed_dim, target_doc_len, target_sent_len, doc_as_sent=False):
        super(DataHelperHotelAll, self).__init__(embed_dim=embed_dim, target_doc_len=target_doc_len,
                                                 target_sent_len=target_sent_len)

        logging.info("setting: %s is %s", "doc_as_sent", doc_as_sent)
        self.doc_as_sent = doc_as_sent

        self.dataset_dir = self.data_path + 'hotel_balance_LengthFix1_3000per/'
        self.num_of_classes = 5

        self.load_all_data()

    def load_files(self, load_test):
        """
        Loads training data from file
        """
        identity = [[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 1]]

        # Load data from files
        train_count = list(open(self.dataset_dir + self.sent_num_file[load_test], "r").readlines())
        train_count = [int(s) for s in train_count if (len(s) > 0 and s != "\n")]

        train_rating = list(open(self.dataset_dir + self.rating_file[load_test], "r").readlines())
        train_rating = [s for s in train_rating if (len(s) > 0 and s != "\n")]
        y = [[identity[int(i) - 1] for i in s.split(" ") if (len(i) > 0 and i != "\n")] for s in train_rating]

        train_content = list(open(self.dataset_dir + self.content_file[load_test], "r").readlines())
        train_content = [s.strip() for s in train_content]

        # Split by words
        x_text = [clean_str(sent) for sent in train_content]

        sentence_lens = []
        x = []
        train_line_index = 0
        for co in train_count:
            review = []
            for currentIndex in range(co):
                tokens = x_text[train_line_index].split()

                if len(tokens) > sent_length_target:  # if longer than limit
                    print(str(len(tokens)) + "\t" + x_text[train_line_index])
                    x_text[train_line_index] = x_text[train_line_index].replace("\?", "")
                    x_text[train_line_index] = x_text[train_line_index].replace("(", "")
                    x_text[train_line_index] = x_text[train_line_index].replace(")", "")
                    x_text[train_line_index] = re.sub(r'[0-9]', '', x_text[train_line_index])
                    tokens = x_text[train_line_index].split()
                    print("\t" + str(len(tokens)) + "\t" + x_text[train_line_index])
                sentence_lens.append(len(tokens))
                review.append(tokens)
                train_line_index += 1
            x.append(review)
        # x_text = [s.split(" ") for s in x_text]

        return [x, y, train_count]

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
    a = DataHelperHotelAll(embed_dim=300, target_doc_len=200, target_sent_len=1024, doc_as_sent=True)
