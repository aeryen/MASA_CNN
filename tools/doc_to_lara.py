import numpy as np
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer

def _dump_svmlight(X, y, f, one_based, query_id):
    is_sp = int(hasattr(X, "tocsr"))
    if X.dtype.kind == 'i':
        value_pattern = "%d:%d"
    else:
        value_pattern = "%d:%.16g"

    line_pattern = ""
    if query_id is not None:
        line_pattern += "qid:%d "
    line_pattern += "%s\n"

    data_id = 1
    for i in range(X.shape[0]):
        if is_sp:
            span = slice(X.indptr[i], X.indptr[i + 1])
            row = zip(X.indices[span], X.data[span])
        else:
            nz = X[i] != 0
            row = zip(np.where(nz)[0], X[i, nz])

        s = " ".join(value_pattern % (j + one_based, x) for j, x in row)

        labels_str = y[i]

        if query_id is not None:
            feat = query_id[i], s
        else:
            feat = s

        f.write("ID" + str(data_id) + "\t" + labels_str.replace(" ", "\t") + "\n")
        for j in range(4):
            f.write((line_pattern % feat).encode('ascii'))
        data_id += 1

train_file = list(open("../data/beer_wholeDoc/" + "train.txt", "r").readlines())
train_content = [sent.strip().lower() for sent in train_file]

test_file = list(open("../data/beer_wholeDoc/" + "test.txt", "r").readlines())
test_content = [sent.strip().lower() for sent in test_file]

train_content.extend(test_content)

count_vect = CountVectorizer(max_features=20000).fit(train_content)
X_train_counts = count_vect.transform(train_content)
print X_train_counts.shape

# =================================================================================

train_rating_file = list(open("../data/beer_wholeDoc/" + "train.rating", "r").readlines())
train_rating = [line.strip() for line in train_rating_file]

test_rating_file = list(open("../data/beer_wholeDoc/" + "test.rating", "r").readlines())
test_rating = [line.strip() for line in test_rating_file]

train_rating.extend(test_rating)

with open('output.txt', 'w') as f:
    _dump_svmlight(X_train_counts, train_rating, f, one_based=False, query_id=None)

