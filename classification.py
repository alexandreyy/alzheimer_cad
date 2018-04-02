'''
Created on 26 de fev de 2016

@author: Alexandre Yukio Yamashita
'''

from sklearn import svm
import numpy as np
from models.performance import compute_performance_metrics, compute_confusion_matrix
from models.k_fold import Kfold
from models.selection_anova import SelectKBestAnova


def save(data, path):
    '''
    Save results.
    '''

    f = file(path, "wb")
    np.save(f, np.array(data))
    f.close()


if __name__ == "__main__":
    '''
    Train and test SVM classifier.
    '''

    # Initialize performance and data vectors.
    accuracies = []
    precisions = []
    recalls = []
    specificities = []
    val_accuracy = dict()
    test_accuracy = dict()
    test_precision = dict()
    test_recall = dict()
    test_specificity = dict()
    features_paths = []

    # Configure database.
    features_selected_limit = 50000
    k_fold_path = "resources/k_fold_10_risk.bin"
    save_performance_path = "resources/performance_risk.np"
    database_path = "resources/dataset_features"
    features_paths.append(database_path + "/features_3_1/")
    features_paths.append(database_path + "/features_5_1/")
    features_paths.append(database_path + "/features_7_1/")
    features_paths.append(database_path + "/features_9_1/")
    selection_path = "scores_risk/"

    k_fold = Kfold()
    k_fold.load(k_fold_path)

    # For all folds, train and test classifier.
    for fold_index in range(len(k_fold.fold_itens)):
        scores = None
        X_train = None
        X_val = None
        X_test = None

        # Load data and scores.
        for path in features_paths:
            k_fold = Kfold()
            k_fold.load(k_fold_path)
            trX_load, y_train, teX_load, y_test, tvX_load, y_val = k_fold.get_train_test_validation_data(path,
                                                                                                         fold_index=fold_index)
            del k_fold
            selection = SelectKBestAnova()
            selection.load(path + selection_path +
                           "fold_" + str(fold_index) + "_risk.np")

            # Get images from AD and Normal subjects.
            trX_load = trX_load[np.where(np.not_equal(y_train, 2))]
            y_train = y_train[np.where(np.not_equal(y_train, 2))]
            trX_load = trX_load[np.where(np.not_equal(y_train, 3))]
            y_train = y_train[np.where(np.not_equal(y_train, 3))]
            teX_load = teX_load[np.where(np.not_equal(y_test, 2))]
            y_test = y_test[np.where(np.not_equal(y_test, 2))]
            teX_load = teX_load[np.where(np.not_equal(y_test, 3))]
            y_test = y_test[np.where(np.not_equal(y_test, 3))]
            tvX_load = tvX_load[np.where(np.not_equal(y_val, 2))]
            y_val = y_val[np.where(np.not_equal(y_val, 2))]
            tvX_load = tvX_load[np.where(np.not_equal(y_val, 3))]
            y_val = y_val[np.where(np.not_equal(y_val, 3))]

            # Sort arg scores to select the same amount of data for each
            # feature.
            scores_tmp = selection.scores_.argsort().argsort()
            selection.scores_ = scores_tmp

            if len(scores_tmp) > features_selected_limit:
                trX_load = selection.transform(
                    trX_load, features_selected_limit)
                teX_load = selection.transform(
                    teX_load, features_selected_limit)
                tvX_load = selection.transform(
                    tvX_load, features_selected_limit)
                scores_tmp = selection.transform(scores_tmp.reshape(
                    (1, len(scores_tmp))), features_selected_limit)[0]

            # Merge data.
            if scores is None:
                scores = scores_tmp
            else:
                scores = np.hstack((scores, scores_tmp))

            if X_train is None:
                X_train = trX_load
                del trX_load
                X_test = teX_load
                del teX_load
                X_val = tvX_load
                del tvX_load
            else:
                X_train = np.hstack((X_train, trX_load))
                del trX_load
                X_test = np.hstack((X_test, teX_load))
                del teX_load
                X_val = np.hstack((X_val, tvX_load))
                del tvX_load

        # Train classifier with grid search to choose the number of features.
        for f in range(1000, features_selected_limit + 1, 1000):
            if f not in val_accuracy.keys():
                val_accuracy[f] = []
                test_accuracy[f] = []
                test_precision[f] = []
                test_recall[f] = []
                test_specificity[f] = []

            selection.scores_ = scores
            X_val_tmp = selection.transform(X_val, f)
            X_train_tmp = selection.transform(X_train, f)
            X_test_tmp = selection.transform(X_test, f)

            # Train and test classifer with validation data.
            print "Training SVM with " + str(f) + " features."
            classifier = svm.SVC(kernel='linear')
            classifier.fit(X_train_tmp, y_train)
            results = compute_confusion_matrix(
                y_val, classifier.predict(X_val_tmp))
            accuracy, precision, recall, specificity = compute_performance_metrics(
                results)
            print fold_index

            # Train and test classifier with test data.
            accuracy_val = accuracy
            classifier.fit(np.vstack((X_train_tmp, X_val_tmp)),
                           np.hstack((y_train, y_val)))
            results = compute_confusion_matrix(
                y_test, classifier.predict(X_test_tmp))
            accuracy, precision, recall, specificity = compute_performance_metrics(
                results)
            val_accuracy[f].append(accuracy_val)
            test_accuracy[f].append(accuracy)
            test_precision[f].append(precision)
            test_recall[f].append(recall)
            test_specificity[f].append(specificity)
            print fold_index

            # Display performance metrics.
            for key in sorted(val_accuracy):
                print key, "-", round(np.mean(np.array(val_accuracy[key]) * 100.0), 1), "-", \
                    round(np.mean(np.array(test_accuracy[key]) * 100.0), 1), "+-", \
                    round(np.std(np.array(test_accuracy[key]) * 100.0), 1), "-", \
                    round(np.mean(np.array(test_precision[key]) * 100.0), 1), "+-", \
                    round(np.std(np.array(test_precision[key]) * 100.0), 1), "-", \
                    round(np.mean(np.array(test_recall[key]) * 100.0), 1), "+-", \
                    round(np.std(np.array(test_recall[key]) * 100.0), 1), "-", \
                    round(np.mean(np.array(test_specificity[key]) * 100.0), 1), "+-", \
                    round(np.std(np.array(test_specificity[key]) * 100.0), 1)

            # Save performance metrics.
            save([val_accuracy, test_accuracy,
                  test_precision, test_recall, test_specificity],
                 save_performance_path)

            # Delete data from memory.
            del X_val_tmp
            del X_train_tmp
            del X_test_tmp
