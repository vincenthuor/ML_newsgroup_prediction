
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
from sklearn import neighbors
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import TruncatedSVD

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets


def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',shuffle=True, remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',shuffle=True, remove=('headers', 'footers', 'quotes'))

    # newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    # newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))
    class_names = newsgroups_train.target_names

    return newsgroups_train, newsgroups_test, class_names

def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names

def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    # evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model

def randomForest(train_bow_tf_idf, train_labels, bow_test_tf_idf, test_labels):
    model = AdaBoostClassifier(n_estimators=100)
    model.fit(train_bow_tf_idf, train_labels)

    print()
    print('------- Random Forest -------')
    # evaluate the model
    print('Default hyperparameters:')
    print(model.get_params())
    train_pred = model.predict(train_bow_tf_idf)
    print('Random Forest train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test_tf_idf)
    print('Random Forest test accuracy = {}'.format((test_pred == test_labels).mean()))
    return model

def Multi_NB(train_bow_tf_idf, train_labels, bow_test_tf_idf, test_labels):
    # training the Multinomial_NB model
    model = MultinomialNB(alpha=0.015)
    model.fit(train_bow_tf_idf, train_labels)

    print()
    print('------- Multinomial Naive Bayes -------')
    # evaluate the model
    print('Default hyperparameters:')
    print(model.get_params())
    train_pred = model.predict(train_bow_tf_idf)
    print('Multinomial NB train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test_tf_idf)
    print('Multinomial NB test accuracy = {}'.format((test_pred == test_labels).mean()))

    # # gridsearch for best Hyperparameter
    # parameters = {'alpha': (1, 0.1, 0.01, 0.015, 0.001)}
    # gs_clf = GridSearchCV(model, parameters, n_jobs=-1)
    # gs_clf = gs_clf.fit(train_bow_tf_idf, train_labels)
    #
    # best_parameters = gs_clf.best_estimator_.get_params()
    # print('Best params using gridSearch:')
    # print(best_parameters)
    # gstrain_pred = gs_clf.predict(train_bow_tf_idf)
    # print('New hyperparameters Multinomial NB train accuracy = {}'.format((gstrain_pred == train_labels).mean()))
    # gstest_pred = gs_clf.predict(bow_test_tf_idf)
    # print('New hyperparameters Multinomial NB test accuracy = {}'.format((gstest_pred == test_labels).mean()))
    # print('---------------------------------------')
    # print()

    return model, test_pred

def Guassian_NB(train_bow_tf_idf, train_labels, bow_test_tf_idf, test_labels):
    # training the Gaussian_NB model
    model = GaussianNB()
    model.fit(train_bow_tf_idf.toarray(), train_labels)

    print()
    print('------- Gaussian Naive Bayes -------')
    # evaluate the model
    print('Default hyperparameters:')
    print(model.get_params())
    train_pred = model.predict(train_bow_tf_idf.toarray())
    print('Gaussian NB train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test_tf_idf)
    print('Gaussian NB test accuracy = {}'.format((test_pred == test_labels).mean()))

    # # gridsearch for best Hyperparameter
    # parameters = {'alpha': (1, 0.1, 0.01, 0.015, 0.001)}
    # gs_clf = GridSearchCV(model, parameters, n_jobs=-1)
    # gs_clf = gs_clf.fit(train_bow_tf_idf, train_labels)
    #
    # best_parameters = gs_clf.best_estimator_.get_params()
    # print('Best params using gridSearch:')
    # print(best_parameters)
    # gstrain_pred = gs_clf.predict(train_bow_tf_idf)
    # print('New hyperparameters Gaussian NB train accuracy = {}'.format((gstrain_pred == train_labels).mean()))
    # gstest_pred = gs_clf.predict(bow_test_tf_idf)
    # print('New hyperparameters Gaussian NB test accuracy = {}'.format((gstest_pred == test_labels).mean()))
    # print('---------------------------------------')
    # print()

    return model

def SVM(train_bow_tf_idf, train_labels, bow_test_tf_idf, test_labels):
    # training the support vector machine (SVM) model. Linear classifiers (SVM) with SGD training

    model = SGDClassifier(loss='squared_hinge', average=100, penalty='l2', alpha=0.0001, random_state=None, max_iter=100, tol=None, n_jobs=-1)
    model.fit(train_bow_tf_idf, train_labels)

    print()
    print('------- Support Vector Machine (SVM) -------')
    # evaluate the model
    print('Default hyperparameters:')
    print(model.get_params())
    train_pred = model.predict(train_bow_tf_idf)
    print('SVM train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test_tf_idf)
    print('SVM test accuracy = {}'.format((test_pred == test_labels).mean()))

    # # gridsearch for best Hyperparameter
    # parameters = {'alpha': (1, 0.1, 0.01, 0.001, 0.0001 ),
    #               'loss': ('squared_hinge', 'hinge' )
    #               }
    # gs_clf = GridSearchCV(model, parameters, n_jobs=-1)
    # gs_clf = gs_clf.fit(train_bow_tf_idf, train_data.target)
    #
    # best_parameters = gs_clf.best_estimator_.get_params()
    # print('Best params using gridSearch:')
    # print(best_parameters)
    # gstrain_pred = gs_clf.predict(train_bow_tf_idf)
    # print('New hyperparameters SVM train accuracy = {}'.format((gstrain_pred == train_labels).mean()))
    # gstest_pred = gs_clf.predict(bow_test_tf_idf)
    # print('New hyperparameters SVM test accuracy = {}'.format((gstest_pred == test_labels).mean()))
    # print('---------------------------------------')
    # print()

    return model, test_pred

def SVM2(train_bow_tf_idf, train_labels, bow_test_tf_idf, test_labels):
    # training the support vector machine (SVM) model. Linear classifiers (SVM) with SGD training

    model = svm.SVC(kernel='poly', degree=3)
    model.fit(train_bow_tf_idf, train_labels)

    print()
    print('------- Support Vector Machine 2 Polynomial Degree 3 (SVM) -------')
    # evaluate the model
    print('Default hyperparameters:')
    print(model.get_params())
    train_pred = model.predict(train_bow_tf_idf)
    print('SVM train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test_tf_idf)
    print('SVM test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model


def LR(train_bow_tf_idf, train_labels, bow_test_tf_idf, test_labels):
    # training Logistic Regression Classifier model

    LR = linear_model.LogisticRegression()
    # model = LR.fit(train_bow_tf_idf, train_labels)
    model = LR.fit(train_bow_tf_idf, train_labels)

    print()
    print('------- Logistic Regression Classifier -------')
    # evaluate the model
    print('Default hyperparameters:')
    print(model.get_params())
    train_pred = model.predict(train_bow_tf_idf)
    print('Logistic Regression train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test_tf_idf)
    print('Logistic Regression test accuracy = {}'.format((test_pred == test_labels).mean()))

    # gridsearch for best Hyperparameter
    parameters = {'C': (1, 0.1, 0.01, 0.001)
                  # 'penalty': ('l1', 'l2'),
                  #   'dual': (False, True)
                  }
    gs_clf = GridSearchCV(model, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(train_bow_tf_idf, train_data.target)

    best_parameters = gs_clf.best_estimator_.get_params()
    print('Best params using gridSearch:')
    print(best_parameters)
    gstrain_pred = gs_clf.predict(train_bow_tf_idf)
    print('New hyperparameters Logistic Regression train accuracy = {}'.format((gstrain_pred == train_labels).mean()))
    gstest_pred = gs_clf.predict(bow_test_tf_idf)
    print('New hyperparameters Logistic Regression test accuracy = {}'.format((gstest_pred == test_labels).mean()))
    print('---------------------------------------')
    print()

def KNN(train_bow_tf_idf, train_labels, bow_test_tf_idf, test_labels, K):
    # training KNN Classifier model

    KNN = neighbors.KNeighborsClassifier(K, weights='distance')
    model = KNN.fit(train_bow_tf_idf, train_data.target)

    print()
    print('------- KNN Classifier -------')
    # evaluate the model
    print('Default hyperparameters:')
    print(model.get_params())
    train_pred = model.predict(train_bow_tf_idf)
    print('KNN Regression train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(bow_test_tf_idf)
    print('KNN Regression test accuracy = {}'.format((test_pred == test_labels).mean()))


def plot_confusion_matrix_color(matrix_c, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    print('Confusion matrix')

    print(matrix_c)

    plt.imshow(matrix_c, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt ='d'
    thresh = matrix_c.max() / 2.
    for i, j in itertools.product(range(matrix_c.shape[0]), range(matrix_c.shape[1])):
        plt.text(j, i, format(matrix_c[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix_c[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    train_data, test_data, class_names = load_data()

    # Count Vectorization
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)

    # TF-idf
    train_bow_tf_idf, test_bow_tf_idf, feature_names_tf_idf = tf_idf_features(train_data, test_data)

    # Baseline Bernoulli Naive Bayes
    bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)
    print(bnb_model)

    # Multinomial NB
    model_MNB ,test_pred  = Multi_NB(train_bow_tf_idf, train_data.target, test_bow_tf_idf, test_data.target)

    # Gaussian NB
    # model_MNB = Guassian_NB(train_bow_tf_idf, train_data.target, test_bow_tf_idf, test_data.target)

    # RandomForest
    # model_RF = randomForest(train_bow_tf_idf, train_data.target, test_bow_tf_idf, test_data.target)

    # SVM Kernel 3rd Degree Polynomial
    # model_SVM2 = SVM2(train_bow_tf_idf, train_data.target, test_bow_tf_idf, test_data.target)

    # Logistic Regression
    model_LR = LR(train_bow_tf_idf, train_data.target, test_bow_tf_idf, test_data.target)

    # KNN
    # model_LR = KNN(train_bow_tf_idf, train_data.target, test_bow_tf_idf, test_data.target, 10)

    # SVM Linear SGD
    model_SVM, test_pred = SVM(train_bow_tf_idf, train_data.target, test_bow_tf_idf, test_data.target)

    ### Compute confusion matrix for SVM Linear SGD ###
    cnf_matrix = confusion_matrix(test_data.target, test_pred)
    np.set_printoptions(precision=3)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix_color(cnf_matrix, classes=class_names,
                          title='Confusion matrix')

    plt.show()

