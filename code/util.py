import numpy as np
from sklearn.cross_validation import KFold, train_test_split

class Util:
    def __init__(self):
        pass

    #Helper method to automatically calculate accuracy given a classifier, nfolds, features and labels
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