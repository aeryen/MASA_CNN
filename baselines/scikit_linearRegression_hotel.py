from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import linear_model
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
import numpy as np
from sklearn import metrics
from sklearn import svm

# categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

file_train_rate = open("../data/hotelreview_balance_3000per/aspect_0.rating", "r").readlines()
# file_train_rate = np.array([np.array(s.split()) for s in file_train_rate])
file_train_rate = [s.split() for s in file_train_rate]
file_train_rate = [[float(num) for num in row] for row in file_train_rate]
file_train_rate = np.array(file_train_rate)

file_train_txt = open("../data/hotelreview_balance_3000per/aspect_0.txt", "r").readlines()

print len(file_train_txt)

count_vect = CountVectorizer(max_features=20000).fit(file_train_txt)
X_train_counts = count_vect.transform(file_train_txt)
print X_train_counts.shape

tf_transformer = TfidfTransformer()
X_train_tf = tf_transformer.fit_transform(X_train_counts)
print X_train_tf.shape

for i in range(5):
    svm_file = open("./hotel_data_1g/train_1g_aspect_" + str(i) + ".txt", mode="aw")
    dump_svmlight_file(X=X_train_counts, y=file_train_rate[:, i], f=svm_file)

"""========================================================================"""

file_test_rate = open("../data/hotelreview_balance_3000per/test_aspect_0.rating", "r").readlines()
file_test_rate = [s.split() for s in file_test_rate]
file_test_rate = [[int(num) for num in row] for row in file_test_rate]
file_test_rate = np.array(file_test_rate)

file_test_txt = open("../data/hotelreview_balance_3000per/test_aspect_0.txt", "r").readlines()

count_vect_test = CountVectorizer(vocabulary=count_vect.vocabulary_, ngram_range=(1, 3))
X_test_counts = count_vect_test.fit_transform(file_test_txt)
print X_test_counts.shape

tf_transformer_test = TfidfTransformer()
X_test_tf = tf_transformer_test.fit_transform(X_test_counts)
print X_test_tf.shape

for i in range(5):
    svm_file = open("./hotel_data_123g/test_123g_aspect_" + str(i) + ".txt", mode="aw")
    dump_svmlight_file(X=X_test_counts, y=file_test_rate[:, i], f=svm_file)

"""========================================================================"""

num_of_aspect = 6

regr = linear_model.LinearRegression()
for i in range(num_of_aspect):
    regr.fit(X_train_counts, file_train_rate[:, i])
    prediction_value = regr.predict(X_test_counts)

    mse = metrics.mean_squared_error(file_test_rate[:, i], prediction_value)
    print str(i) + " MSE: \t" + str(mse)

    r2 = metrics.r2_score(file_test_rate[:, i], prediction_value)
    print str(i) + " R2 : \t" + str(r2)

    prediction_value_int = np.rint(prediction_value)
    acc = metrics.accuracy_score(file_test_rate[:, i], prediction_value_int)
    print str(i) + " ACC : \t" + str(acc)
#
regr = linear_model.Ridge()
for i in range(num_of_aspect):
    regr.fit(X_train_counts, file_train_rate[:, i])
    prediction_value = regr.predict(X_test_counts)

    mse = metrics.mean_squared_error(file_test_rate[:, i], prediction_value)
    print str(i) + " MSE: \t" + str(mse)

    r2 = metrics.r2_score(file_test_rate[:, i], prediction_value)
    print str(i) + " R2 : \t" + str(r2)

    prediction_value_int = np.rint(prediction_value)
    acc = metrics.accuracy_score(file_test_rate[:, i], prediction_value_int)
    print str(i) + " ACC : \t" + str(acc)
#
regr = linear_model.Lasso()
for i in range(num_of_aspect):
    regr.fit(X_train_counts, file_train_rate[:, i])
    prediction_value = regr.predict(X_test_counts)

    mse = metrics.mean_squared_error(file_test_rate[:, i], prediction_value)
    print str(i) + " MSE: \t" + str(mse)

    r2 = metrics.r2_score(file_test_rate[:, i], prediction_value)
    print str(i) + " R2 : \t" + str(r2)

    prediction_value_int = np.rint(prediction_value)
    acc = metrics.accuracy_score(file_test_rate[:, i], prediction_value_int)
    print str(i) + " ACC : \t" + str(acc)

clf = svm.SVR(kernel="linear", max_iter=30000, cache_size=500)
for i in range(num_of_aspect):
    clf.fit(X_train_counts, file_train_rate[:, i])
    prediction_value = clf.predict(X_test_counts)

    mse = metrics.mean_squared_error(file_test_rate[:, i], prediction_value)
    print str(i) + " MSE: \t" + str(mse)

    r2 = metrics.r2_score(file_test_rate[:, i], prediction_value)
    print str(i) + " R2 : \t" + str(r2)

    prediction_value_int = np.rint(prediction_value)
    acc = metrics.accuracy_score(file_test_rate[:, i], prediction_value_int)
    print str(i) + " ACC : \t" + str(acc)

clf = svm.SVC(kernel='linear', max_iter=30000, cache_size=500)
for i in range(num_of_aspect):
    clf.fit(X_train_counts, file_train_rate[:, i])
    prediction_value = clf.predict(X_test_counts)

    mse = metrics.mean_squared_error(file_test_rate[:, i], prediction_value)
    print str(i) + " MSE: \t" + str(mse)

    r2 = metrics.r2_score(file_test_rate[:, i], prediction_value)
    print str(i) + " R2 : \t" + str(r2)

    prediction_value_int = np.rint(prediction_value)
    acc = metrics.accuracy_score(file_test_rate[:, i], prediction_value_int)
    print str(i) + " ACC : \t" + str(acc)
