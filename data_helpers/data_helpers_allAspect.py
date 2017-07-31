import numpy as np
import re
import itertools
import pickle
from collections import Counter


def clean_str(string):
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


def load_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    identity = [[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]]

    # Load data from files
    train_count = list(open("./data/BalanceByAspect0_LengthFix1_3000perRate/aspect_0.count", "r").readlines())
    train_count = [int(s) for s in train_count if (len(s) > 0 and s != "\n")]

    train_rating = list(open("./data/BalanceByAspect0_LengthFix1_3000perRate/aspect_0.rating", "r").readlines())
    train_rating = [s for s in train_rating if (len(s) > 0 and s != "\n")]
    y = [[identity[int(i) - 1] for i in s.split(" ") if (len(i) > 0 and i != "\n")] for s in train_rating]

    train_content = list(open("./data/BalanceByAspect0_LengthFix1_3000perRate/aspect_0.txt", "r").readlines())
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

            if len(tokens) > 65:
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

    return [x, y]


def load_test_data_and_labels():
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    identity = [[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]];


    # Load data from files
    train_count = list(open("./data/BalanceByAspect0_LengthFix1_3000perRate/test_aspect_0.count", "r").readlines())
    train_count = [int(s) for s in train_count if (len(s) > 0 and s != "\n")]

    train_rating = list(open("./data/BalanceByAspect0_LengthFix1_3000perRate/test_aspect_0.rating", "r").readlines())
    train_rating = [s for s in train_rating if (len(s) > 0 and s != "\n")]
    y = [[identity[int(i) - 1] for i in s.split(" ") if (len(i) > 0 and i != "\n")] for s in train_rating]

    train_content = list(open("./data/BalanceByAspect0_LengthFix1_3000perRate/test_aspect_0.txt", "r").readlines())
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

            if len(tokens) > 65:
                print(str(len(tokens)) + "\t" + x_text[train_line_index])
                x_text[train_line_index] = x_text[train_line_index].replace("\?", "")
                x_text[train_line_index] = x_text[train_line_index].replace("(", "")
                x_text[train_line_index] = x_text[train_line_index].replace(")", "")
                x_text[train_line_index] = re.sub(r'[0-9]', '', x_text[train_line_index])
                tokens = x_text[train_line_index].split()
                if len(tokens) > 67:
                    tokens = tokens[:67]
                print("\t" + str(len(tokens)) + "\t" + x_text[train_line_index])
            sentence_lens.append(len(tokens))
            review.append(tokens)
            train_line_index += 1
        x.append(review)
    # x_text = [s.split(" ") for s in x_text]

    return [x, y, train_count]


def pad_sentences(review_all, y, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    review_sent_lengths = [[len(sent) for sent in review] for review in review_all]
    max_length = max([max(review_length) for review_length in review_sent_lengths])

    # long_sentence = sentence_lengths.index(sequence_length)
    # print "longest sequence length: " + str(sequence_length)
    # print reviews[long_sentence]

    padded_reviews = []
    for review_i in range(len(review_all)):
        review = review_all[review_i]
        padded_sentences = []
        for sent in review:
            num_padding = max_length - len(sent)
            new_sentence = sent + [padding_word] * num_padding
            padded_sentences.append(new_sentence)
        padded_reviews.append(padded_sentences)
    return [padded_reviews, y]


def build_vocab(reviews):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    unrolled_words = [val for sublist in reviews for val in sublist]
    word_counts = Counter(itertools.chain(*unrolled_words))
    # Mapping from index to word
    # TODO limit vocab size
    vocabulary_inv = [x[0] for x in word_counts.most_common()]

    print("size of vocabulary: " + str(len(vocabulary_inv)))
    # vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary_inv = list(vocabulary_inv[:20000])  # limit vocab size

    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    return [vocabulary, vocabulary_inv]


def build_input_data(reviews, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    pad_value = vocabulary["<PAD/>"]
    max_channel_size = 100

    value_tensor = [[[vocabulary.get(word, pad_value)
                      for word in sentence]
                     for sentence in review]
                    for review in reviews]

    padded_sentence_length = len(reviews[0][0])
    empty_channel = [pad_value] * padded_sentence_length
    reviews = [review[:max_channel_size] if len(review) > max_channel_size else review for review in value_tensor]

    padded_review = []
    for review_i in range(len(reviews)):
        num_padding = max_channel_size - len(reviews[review_i])
        new_review = reviews[review_i] + [empty_channel] * num_padding
        padded_review.append(new_review)

    x = np.array(padded_review)
    y = np.array(labels)
    return [x, y]


def pad_reviews(x, vocabulary, padding_word="<PAD/>"):
    padded_sentence_length = x.shape[2]
    empty_channel = vocabulary["<PAD/>"] * padded_sentence_length
    max_channel = max([len(review) for review in x])
    padded_review = []
    for review_i in range(len(x)):
        num_padding = max_channel - len(x[review_i])
        new_review = x[review_i] + [empty_channel] * num_padding
        padded_review.append(new_review)
    return padded_review


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded, labels = pad_sentences(sentences, labels)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    pickle.dump([vocabulary, vocabulary_inv], open("vocabulary_full.pickle", "wb"))

    x, y = build_input_data(sentences_padded, labels, vocabulary)

    return [x, y, vocabulary, vocabulary_inv]


def load_test_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, s_count = load_test_data_and_labels()
    sentences_padded, labels = pad_sentences(sentences, labels)
    # vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    vocabulary, vocabulary_inv = pickle.load(open("vocabulary_full.pickle", "rb"))
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, s_count]


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
