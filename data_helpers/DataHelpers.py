import numpy as np
import re
import logging
import pickle
from sklearn.preprocessing import OneHotEncoder
import os
from collections import Counter


class DataHelper(object):
    def __init__(self, embed_dim, target_doc_len, target_sent_len):
        logging.info("setting: %s is %s", "embed_dim", embed_dim)
        logging.info("setting: %s is %s", "target_doc_len", target_doc_len)

        assert embed_dim is not None
        assert target_sent_len is not None

        self.num_of_classes = None

        self.embedding_dim = embed_dim
        self.target_doc_len = target_doc_len
        self.target_sent_len = target_sent_len

        self.train_data = None
        self.test_data = None
        self.vocab = None
        self.vocab_inv = None
        self.embed_matrix = None
        self.vocabulary_size = 20000

        self.glove_dir = os.path.join(os.path.dirname(__file__), 'glove/')
        self.glove_path = self.glove_dir + "glove.6B." + str(self.embedding_dim) + "d.txt"
        self.w2v_dir = os.path.join(os.path.dirname(__file__), 'w2v/')
        self.w2v_path = self.w2v_dir + "GoogleNews-vectors-negative300.bin"

        self.data_path = os.path.join(os.path.dirname(__file__), '..', 'data/')

        print("loading embedding.")
        glove_pickle = os.path.join(os.path.dirname(__file__), 'glove.pickle')
        # [self.glove_words, self.glove_vectors] = self.load_glove_vector()
        # pickle.dump([self.glove_words, self.glove_vectors], open("glove.pickle", "wb"))
        self.glove_words, self.glove_vectors = pickle.load(open(glove_pickle, "rb"))
        print("loading embedding completed.")

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_vocab(self):
        return self.vocab

    def get_vocab_inv(self):
        return self.vocab_inv

    def clean_str(self, string):
        # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub("\'", " \' ", string)
        string = re.sub("\"", " \" ", string)
        string = re.sub("-", " - ", string)

        string = re.sub(",", " , ", string)
        string = re.sub("\.", " \. ", string)
        string = re.sub("!", " ! ", string)
        string = re.sub("\?", " \? ", string)

        string = re.sub(r"[(\[{]", " ( ", string)
        string = re.sub(r"[)\]}]", " ) ", string)
        string = re.sub("\s{2,}", " ", string)

        return string.strip().lower()

    def load_glove_vector(self):
        glove_lines = list(open(self.glove_path, "r", encoding="utf-8").readlines())
        glove_lines = [s.split(" ", 1) for s in glove_lines if (len(s) > 0 and s != "\n")]
        glove_words = [s[0] for s in glove_lines]
        vector_list = [s[1] for s in glove_lines]
        glove_vectors = np.array([np.fromstring(line, dtype=float, sep=' ') for line in vector_list])
        return [glove_words, glove_vectors]

    def build_glove_embedding(self, vocabulary_inv):
        np.random.seed(10)
        embed_matrix = []
        std = np.std(self.glove_vectors[0, :])
        for word in vocabulary_inv:
            if word in self.glove_words:
                word_index = self.glove_words.index(word)
                embed_matrix.append(self.glove_vectors[word_index, :])
            elif word == "<PAD>":
                embed_matrix.append(np.zeros(self.embedding_dim))
            else:
                embed_matrix.append(np.random.normal(loc=0.0, scale=std, size=self.embedding_dim))
        embed_matrix = np.array(embed_matrix)
        return embed_matrix

    def pad_sentences_word(self, sentences, padding_word="<PAD/>"):
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

    def pad_sentences(self, data):
        if self.target_sent_len is not None and self.target_sent_len > 0:
            max_length = self.target_sent_len
        else:
            sent_lengths = [[len(sent) for sent in doc] for doc in data.value]
            max_length = max(sent_lengths)
            print(("longest doc: " + str(max_length)))

        padded_sents = []
        for sent in data.value:
            if len(sent) <= max_length:
                num_padding = max_length - len(sent)
                new_sentence = np.concatenate([sent, np.zeros(num_padding, dtype=np.int)])
            else:
                new_sentence = sent[:max_length]

            padded_sents.append(new_sentence)
        data.value = np.array(padded_sents)
        return data

    @staticmethod
    def pad_document(data, target_length=-1):
        docs = data.value
        lens = data.doc_size
        if target_length > 0:
            tar_length = target_length
        else:
            tar_length = max(lens)
            print("longest doc: " + str(tar_length))

        padded_doc = []
        trim_len = []
        sent_length = len(docs[0][0])
        for i in range(len(docs)):
            d = docs[i]
            if len(d) <= tar_length:
                num_padding = tar_length - len(d)
                if len(d) > 0:
                    new_doc = np.concatenate([d, np.zeros([num_padding, sent_length], dtype=np.int)])
                    trim_len.append(lens[i])
                else:
                    raise ValueError("Warning, 0 line file!")
            else:
                new_doc = d[:tar_length]
                trim_len.append(tar_length)
            padded_doc.append(new_doc)
        data.value = np.array(padded_doc)
        data.doc_size_trim = np.array(trim_len)
        return data

    @staticmethod
    def concat_to_doc(sent_list, sent_count):
        start_index = 0
        docs = []
        for s in sent_count:
            doc = " <LB> ".join(sent_list[start_index:start_index+s])
            docs.append(doc)
            start_index = start_index + s
        return docs

    @staticmethod
    def chain(data_splits):
        for data in data_splits:
            for sent in data.raw:
                for word in sent:
                    yield word

    @staticmethod
    def build_vocab(data, vocabulary_size):
        # Build vocabulary
        word_counts = Counter(DataHelper.chain(data))
        word_counts = sorted(list(word_counts.items()), key=lambda t: t[::-1], reverse=True)
        vocabulary_inv = [item[0] for item in word_counts]
        vocabulary_inv.insert(0, "<PAD>")
        vocabulary_inv.insert(1, "<UNK>")

        logging.info("size of vocabulary: " + str(len(vocabulary_inv)))
        vocabulary_inv = list(vocabulary_inv[:vocabulary_size])  # limit vocab size

        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    @staticmethod
    def to_onehot(label_vector, total_class):
        y_onehot = np.zeros((len(label_vector), total_class))
        y_onehot[np.arange(len(label_vector)), label_vector.astype(int)] = 1
        return y_onehot

    @staticmethod
    def to_onehot_3d(label_vector, total_class):
        label_vector .astype(np.int)
        y_onehot = np.zeros((len(label_vector), len(label_vector[0]), total_class))
        for instance_index in range(len(label_vector)):
            for aspect_index in range(len(label_vector[0])):
                y_onehot[instance_index][aspect_index][label_vector[instance_index][aspect_index]] = 1

        return y_onehot

    def build_content_vector(self, data):
        unk = self.vocab["<UNK>"]
        content_vector = np.array([[self.vocab.get(word, unk) for word in sent] for sent in data.raw])
        data.value = content_vector
        return data

    @staticmethod
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
