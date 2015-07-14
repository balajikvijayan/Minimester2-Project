import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold, train_test_split,cross_val_score

class Util:
    def __init__(self):
        pass

    #Helper method to automatically calculate accuracy given a classifier, nfolds, features and labels
    #TO BE DEPRECATED
    def CalculateAccuracy(self, clf, nfolds, train_features, train_labels, test_features, test_labels):
        #Kfold and accuracy initialization
        kf = KFold(train_features.shape[0], n_folds = nfolds)
        train_accuracy, test_accuracy = np.empty(nfolds), np.empty(nfolds)
        #Need a less shitty way of enumerating through the kfold loop
        i = 0
        #Loop through Kfolds to calculate accuracy
        for train_index, test_index in kf:
            training_features = train_features[train_index]
            traintest_features = train_features[test_index]
            
            training_labels = train_labels[train_index]
            traintest_labels = train_labels[test_index]
            
            clf.fit(training_features, training_labels)
            train_accuracy[i] = clf.score(traintest_features, traintest_labels)
            test_accuracy[i] = clf.score(test_features, test_labels)
            i+=1
        return np.mean(train_accuracy), np.mean(test_accuracy)
    
    #Given a list of classifiers (hyperparameter tuned), X, y, cv size and scoring method, return a score list and a time list
    def TimevScore(self, clf_list, X, y, k, score_str):
        scores, times = [],[]
        for clf in clf_list:
            time0 = time.time()
            scores.append(cross_val_score(clf, X, y,cv=k, scoring=score_str))
            times.append(time.time()-time0)
        return times,np.mean(scores,axis=1)