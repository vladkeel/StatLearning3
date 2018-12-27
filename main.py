import numpy as np
import pandas as pd
from sklearn import svm
from random import shuffle

from utils import load_mnist
from sklearn.model_selection import train_test_split


def cv(X, y, model, folds):
    """

    :param X: Training set data
    :param y: Training set labels
    :param model: The classifier, initialized outside the function
    :param folds: int - How many K-folds for cross validation
    :return: tuple of floats (t, v) where t is training error and v is validation error
    """
    fold_size = np.floor(len(X) / folds)
    indices = [x for x in range(len(X))]
    shuffle(indices)
    # indices is a set of indices corresponding to entries in training data
    # shuffled to random order.
    new_x = np.array(X)[indices]  # Shuffled training set data
    new_y = np.array(y)[indices]  # Shuffled training set labels
    foldss = [(int(x * fold_size), int(x * fold_size + fold_size)) for x in range(folds - 1)]
    # foldss is a list of tuples indicating the start and finish point of folds
    foldss.append((int((folds - 1) * fold_size), int(-1)))
    results_val = []
    results_train = []
    for i in range(folds):
        train_x = np.concatenate((new_x[:foldss[i][0]], new_x[foldss[i][1]:]))
        train_y = np.concatenate((new_y[:foldss[i][0]], new_y[foldss[i][1]:]))
        # train data is all folds except for fold i
        test_x = new_x[foldss[i][0]:foldss[i][1]]
        test_y = new_y[foldss[i][0]:foldss[i][1]]
        # test data is fold i
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        # calculate cross validation error
        res_val = sum([test_y[i] != pred_y[i] for i in range(len(pred_y))]) / len(pred_y)
        results_val.append(res_val)
        pred_test = model.predict(train_x)
        # calculate training error
        res_train = sum([train_y[i] != pred_test[i] for i in range(len(pred_y))]) / len(pred_test)
        results_train.append(res_train)
    return np.average(results_train), np.average(results_val)


if __name__ == '__main__':
    data_df, labels_df = load_mnist()
    data_df = np.array(data_df)
    labels_df = np.array(labels_df)
    # Normalizing data to [-1,1] to cut down training and inference time.
    data_df = data_df / 225.0 * 2 - 1
    X_train, X_test, y_train, y_test = train_test_split(data_df, labels_df, random_state=98, test_size=0.2)
    lables = ['linear', 'poly', 'rbf']
    gamma = [0.001, 0.01, 0.1, 1, 10]
    degree = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for lable in lables:
        iteritems = None
        if lable == 'linear':
            iteritems = [1]
        elif lable == 'poly':
            iteritems = degree
        else:
            iteritems = gamma
        for parameter in iteritems:
            clf = None
            if lable == 'linear':
                clf = svm.SVC(kernel=lable)
            elif lable == 'poly':
                clf = svm.SVC(kernel=lable, degree=parameter, gamma='auto')
            else:
                clf = svm.SVC(kernel=lable, gamma=parameter)
            train_res, cv_res = cv(X_train, y_train, clf, 5)
            clf.fit(X_train, y_train)
            pred_y2 = clf.predict(X_test)
            test_res = sum([pred_y2[i] != y_test[i] for i in range(len(pred_y2))]) / len(pred_y2)
            if lable == 'linear':
                print("svm_{}: {}, {}, {}".format(lable, train_res, cv_res, test_res))
            else:
                print("svm_{}_{}{}: {}, {}, {}".format(lable, 'g' if lable == 'rbf' else 'd',
                                                       parameter, train_res, cv_res, test_res))
