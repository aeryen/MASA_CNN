import numpy as np
from sklearn.metrics import classification_report

input_dir = "../runs/my_beer/regression2/1482116297_120/"

predict_aspect_list = list(open(input_dir + "aspect_related_name.out", "r").readlines())
predict_aspect_list = [s.strip() for s in predict_aspect_list if (len(s) > 0 and s != "\n")]
predict_aspect_list = np.array(predict_aspect_list)

test_file = list(open("../data/beer_100k/" + "test.txt", "r").readlines())
test_content = [sent.strip().lower() for sent in test_file]

aspect_keywords = [["a:", "a-"], ["t:", "t-"], ["m:", "m-"], ["s:", "s-"]]
aspect_name = ["appearance", "taste", "palate", "aroma"]

total_sent_correct = 0
total_sent_count = 0

sentences_with_key = []
sentences_true_label = []

for sentence_index in range(len(test_content)):
    no_class = True
    for aspect_index in range(len(aspect_keywords)):
        for aspect_key in aspect_keywords[aspect_index]:
            if test_content[sentence_index].startswith(aspect_key):
                sentences_with_key.append(sentence_index)
                sentences_true_label.append(aspect_name[aspect_index])

sentences_true_label = np.array(sentences_true_label)

sentences_with_key = np.array(sentences_with_key)
predict_labels = predict_aspect_list[sentences_with_key]
predict_labels = np.array(predict_labels)

result_yifan = classification_report(sentences_true_label, predict_labels, digits=2)
print result_yifan
