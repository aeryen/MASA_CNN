import numpy as np
import re
import logging
import itertools
import pkg_resources


class DataHelper(object):
    def __init__(self, doc_level, embed_type, embed_dim, target_doc_len, target_sent_len):
        logging.info("setting: %s is %s", "doc_level", doc_level)
        logging.info("setting: %s is %s", "embed_type", embed_type)
        logging.info("setting: %s is %s", "embed_dim", embed_dim)
        logging.info("setting: %s is %s", "target_doc_len", target_doc_len)

        assert doc_level is not None
        assert embed_type is not None
        assert embed_dim is not None
        assert target_sent_len is not None

        self.num_of_classes = None

        self.doc_level_data = doc_level
        self.embed_type = embed_type
        self.embedding_dim = embed_dim
        self.target_doc_len = target_doc_len
        self.target_sent_len = target_sent_len

        self.train_data = None
        self.test_data = None
        self.vocab = None
        self.vocab_inv = None
        self.embed_matrix = None
        self.vocabulary_size = 20000

        self.glove_dir = pkg_resources.resource_filename('datahelpers', 'glove/')
        self.glove_path = self.glove_dir + "glove.6B." + str(self.embedding_dim) + "d.txt"
        self.w2v_dir = pkg_resources.resource_filename('datahelpers', 'w2v/')
        self.w2v_path = self.w2v_dir + "GoogleNews-vectors-negative300.bin"

        if self.embed_type == "glove":
            [self.glove_words, self.glove_vectors] = self.load_glove_vector()

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_vocab(self):
        return self.vocab

    def get_vocab_inv(self):
        return self.vocab_inv

    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def load_glove_vector(self):
        glove_lines = list(open(self.glove_path, "r", encoding="utf-8").readlines())
        glove_lines = [s.split(" ", 1) for s in glove_lines if (len(s) > 0 and s != "\n")]
        glove_words = [s[0] for s in glove_lines]
        vector_list = [s[1] for s in glove_lines]
        glove_vectors = np.array([np.fromstring(line, dtype=float, sep=' ') for line in vector_list])
        return [glove_words, glove_vectors]

    def pad_sentences(self, sentences, padding_word="<PAD/>"):
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
        sequence_length = max(len(x) for x in sentences)
        padded_sentences = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            num_padding = sequence_length - len(sentence)
            new_sentence = sentence + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        return padded_sentences

    def build_vocab(self, sentences):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*sentences))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(vocabulary_inv))
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    def build_input_data(self, sentences, labels, vocabulary):
        """
        Maps sentencs and labels to vectors based on a vocabulary.
        """
        x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
        y = np.array(labels)
        return [x, y]

    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
