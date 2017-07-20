from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model
from sklearn.datasets import dump_svmlight_file
import numpy as np
from sklearn import metrics
from sklearn import svm

# categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)


file_train_rate = open("../data/beer_wholeDoc/train.rating", "r").readlines()
# file_train_rate = np.array([np.array(s.split()) for s in file_train_rate])
file_train_rate = [s.split() for s in file_train_rate]
file_train_rate = [[float(num) for num in row] for row in file_train_rate]
file_train_rate = np.array(file_train_rate)

file_train_txt = open("../data/beer_wholeDoc/train.txt", "r").readlines()

print len(file_train_txt)

count_vect = CountVectorizer(max_features=20000).fit(file_train_txt)
X_train_counts = count_vect.transform(file_train_txt)
print X_train_counts.shape

tf_transformer = TfidfTransformer()
X_train_tf = tf_transformer.fit_transform(X_train_counts)
print X_train_tf.shape

for i in range(5):
    svm_file = open("./beer_data_123g/train_1g_aspect_" + str(i) + ".txt", mode="aw")
    dump_svmlight_file(X=X_train_counts, y=file_train_rate[:, i], f=svm_file)

"""========================================================================"""

file_test_rate = open("../data/beer_wholeDoc/test.rating", "r").readlines()
file_test_rate = [s.split() for s in file_test_rate]
file_test_rate = [[float(num) for num in row] for row in file_test_rate]
file_test_rate = np.array(file_test_rate)

file_test_txt = open("../data/beer_wholeDoc/test.txt", "r").readlines()

count_vect_test = CountVectorizer(vocabulary=count_vect.vocabulary_)
X_test_counts = count_vect_test.fit_transform(file_test_txt)
print X_test_counts.shape

tf_transformer_test = TfidfTransformer()
X_test_tf = tf_transformer_test.fit_transform(X_test_counts)
print X_test_tf.shape

for i in range(5):
    svm_file = open("./beer_data_123g/test_1g_aspect_" + str(i) + ".txt", mode="aw")
    dump_svmlight_file(X=X_test_counts, y=file_test_rate[:, i], f=svm_file)

"""========================================================================"""

regr = linear_model.LinearRegression()

for i in range(5):
    regr.fit(X_train_tf, file_train_rate[:, i])
    prediction_value = regr.predict(X_test_tf)

    mse = metrics.mean_squared_error(file_test_rate[:, i], prediction_value)
    print str(i) + " MSE: \t" + str(mse)

    r2 = metrics.r2_score(file_test_rate[:, i], prediction_value)
    print str(i) + " R2 : \t" + str(r2)

regr = linear_model.Ridge()

for i in range(5):
    regr.fit(X_train_tf, file_train_rate[:, i])
    prediction_value = regr.predict(X_test_tf)

    mse = metrics.mean_squared_error(file_test_rate[:, i], prediction_value)
    print str(i) + " MSE: \t" + str(mse)

    r2 = metrics.r2_score(file_test_rate[:, i], prediction_value)
    print str(i) + " R2 : \t" + str(r2)

regr = linear_model.Lasso()

for i in range(5):
    regr.fit(X_train_tf, file_train_rate[:, i])
    prediction_value = regr.predict(X_test_tf)

    mse = metrics.mean_squared_error(file_test_rate[:, i], prediction_value)
    print str(i) + " MSE: \t" + str(mse)

    r2 = metrics.r2_score(file_test_rate[:, i], prediction_value)
    print str(i) + " R2 : \t" + str(r2)

clf = svm.SVR(kernel="linear", max_iter=20000, cache_size=500)
for i in range(5):
    print "a"
    clf.fit(X_train_tf, file_train_rate[:, i])
    print "b"
    prediction_value = clf.predict(X_test_tf)
    print "c"

    mse = metrics.mean_squared_error(file_test_rate[:, i], prediction_value)
    print str(i) + " MSE: \t" + str(mse)

    r2 = metrics.r2_score(file_test_rate[:, i], prediction_value)
    print str(i) + " R2 : \t" + str(r2)
