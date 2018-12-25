import numpy as np
import pandas as pd
from sklearn import svm
from random import shuffle

from utils import load_mnist
from sklearn.model_selection import train_test_split


def cv(X, y, model, folds):
    fold_size = np.floor(len(X)/folds)
    indices = [x for x in range(len(X))]
    shuffle(indices)
    new_x = np.array(X)[indices]
    new_y = np.array(y)[indices]
    foldss = [(int(x*fold_size), int(x*fold_size + fold_size)) for x in range(folds - 1)]
    foldss.append((int((folds - 1)*fold_size), int(-1)))
    results = []
    for i in range(folds):
        train_x = np.concatenate((new_x[:foldss[i][0]], new_x[foldss[i][1]:]))
        train_y = np.concatenate((new_y[:foldss[i][0]], new_y[foldss[i][1]:]))
        test_x = new_x[foldss[i][0]:foldss[i][1]]
        test_y = new_y[foldss[i][0]:foldss[i][1]]
        model.fit(train_x, train_y)
        pred_y = model.predict(test_x)
        res = sum([test_y[i] != pred_y[i] for i in range(len(pred_y))])/len(pred_y)
        results.append(res)
    return np.average(results)


if __name__ == '__main__':
    data_df, labels_df = load_mnist()
    data_df = np.array(data_df)
    labels_df = np.array(labels_df)
    data_df = data_df / 225.0 * 2 - 1
    X_train, X_test, y_train, y_test = train_test_split(data_df, labels_df, random_state=98, test_size=0.2)
    clf = svm.SVC(kernel='linear')
    lables = ['linear', 'poly', 'rbf']
    gamma = [0.001, 0.01, 0.1, 1, 10]
    degree = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    cv_res = cv(X_train, y_train, clf, 5)
    clf.fit(X_train, y_train)
    pred_y = clf.predict(X_train)
    train_res = sum([pred_y[i] != y_train[i] for i in range(len(pred_y))])/len(pred_y)
    pred_y2 = clf.predict(X_test)
    test_res = sum([pred_y2[i] != y_test[i] for i in range(len(pred_y2))])/len(pred_y2)
    print("svm_{}_{}: {}, {}, {}".format(lables[0], None, cv_res, train_res, test_res))

