import numpy as np
import re
import itertools
import pickle
from collections import Counter


class DataHelperHotelOne(object):
    def load_data_and_labels(aspect_id):
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
        y = [identity[int(s.split(" ")[aspect_id]) - 1] for s in train_rating]

        train_content = list(open("./data/BalanceByAspect0_LengthFix1_3000perRate_Review/aspect_0.txt", "r").readlines())
        train_content = [s.strip() for s in train_content]
        # Split by words
        x_text = [clean_str(sent) for sent in train_content]

        # review_lens = []
        x = []
        for train_line_index in range(len(x_text)):
            tokens = x_text[train_line_index].split()

            if len(tokens) > 5120:
                print str(len(tokens)) + "\t" + x_text[train_line_index]
                x_text[train_line_index] = x_text[train_line_index].replace("\?", "")
                x_text[train_line_index] = x_text[train_line_index].replace("(", "")
                x_text[train_line_index] = x_text[train_line_index].replace(")", "")
                x_text[train_line_index] = re.sub(r'[0-9]', '', x_text[train_line_index])

                tokens = x_text[train_line_index].split()

                if len(tokens) > 6700:
                    tokens = tokens[:6700]
                    print "\t### Force Cut"
                    # print "\t" + str(len(tokens)) + "\t" + x_text[train_line_index]
            x.append(tokens)

        return [x, y]


    def load_test_data_and_labels(aspect_id):
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
        x_text = [clean_str(sent) for sent in train_content]

        # review_lens = []
        x = []
        for train_line_index in range(len(x_text)):
            tokens = x_text[train_line_index].split()

            if len(tokens) > 5120:
                print str(len(tokens)) + "\t" + x_text[train_line_index]
                x_text[train_line_index] = x_text[train_line_index].replace("\?", "")
                x_text[train_line_index] = x_text[train_line_index].replace("(", "")
                x_text[train_line_index] = x_text[train_line_index].replace(")", "")
                x_text[train_line_index] = re.sub(r'[0-9]', '', x_text[train_line_index])

                tokens = x_text[train_line_index].split()

                if len(tokens) > 6700:
                    tokens = tokens[:6700]
                    print "\t### Force Cut"
                    # print "\t" + str(len(tokens)) + "\t" + x_text[train_line_index]
            x.append(tokens)

        return [x, y]


    def pad_sentences(reviews, y, padding_word="<PAD/>", target_length=-1):
        """
        Pads all sentences to the same length. The length is defined by the longest sentence.
        Returns padded sentences.
        """
        if target_length > 0:
            max_length = target_length
        else:
            review_lengths = [len(x) for x in reviews]
            max_length = max(review_lengths)
            long_review_index = review_lengths.index(max_length)
            print "longest sequence length: " + str(max_length)
            print reviews[long_review_index]

        padded_sentences = []
        for i in range(len(reviews)):
            rev = reviews[i]
            num_padding = max_length - len(rev)
            new_sentence = rev + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        return [padded_sentences, y]


    def build_vocab(reviews):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        word_counts = Counter(itertools.chain(*reviews))
        # Mapping from index to word
        vocabulary_inv = [x[0] for x in word_counts.most_common()]

        print "size of vocabulary: " + str(len(vocabulary_inv))
        # vocabulary_inv = list(sorted(vocabulary_inv))
        vocabulary_inv = list(vocabulary_inv[:20000])  # limit vocab size

        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]


    def build_input_data(reviews, labels, vocabulary):
        """
        Maps sentencs and labels to vectors based on a vocabulary.
        """
        x = np.array([[vocabulary.get(word, vocabulary["<PAD/>"]) for word in rev] for rev in reviews])
        y = np.array(labels)
        return [x, y]


    def load_glove_vector():
        glove_lines = list(open("./glove.6B.100d.txt", "r").readlines())
        glove_lines = [s.split() for s in glove_lines if (len(s) > 0 and s != "\n")]
        glove_words = [s[0] for s in glove_lines]
        vector_list = [s[1:] for s in glove_lines]
        glove_vectors = np.array([[float(n) for n in line] for line in vector_list])
        return [glove_words, glove_vectors]


    def build_embedding(vocabulary_inv, glove_words, glove_vectors):
        embed_matrix = []
        std = np.std(glove_vectors[0, :])
        for word in vocabulary_inv:
            if word in glove_words:
                word_index = glove_words.index(word)
                embed_matrix.append(glove_vectors[word_index, :])
            else:
                embed_matrix.append(np.random.normal(loc=0.0, scale=std, size=100))
        embed_matrix = np.array(embed_matrix)
        return embed_matrix


    def load_data(aspect_id):
        """
        Loads and preprocessed data for the MR dataset.
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        sentences, labels = load_data_and_labels(aspect_id=aspect_id)

        sentences_padded, labels = pad_sentences(sentences, labels)
        vocabulary, vocabulary_inv = build_vocab(sentences_padded)
        pickle.dump([vocabulary, vocabulary_inv], open("vocabulary.pickle", "wb"))

        [glove_words, glove_vectors] = load_glove_vector()
        embed_matrix = build_embedding(vocabulary_inv, glove_words, glove_vectors)
        # pickle.dump([embed_matrix], open("embed_matrix.pickle", "wb"))

        x, y = build_input_data(sentences_padded, labels, vocabulary)
        return [x, y, vocabulary, vocabulary_inv, embed_matrix]


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


if __name__ == "__main__":
    x, y, vocabulary, vocabulary_inv = load_data()
