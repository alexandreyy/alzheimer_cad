'''
Created on 26 de fev de 2016

@author: Alexandre Yukio Yamashita
'''

import numpy as np


def load(path):
    '''
    Load results.
    '''

    f = file(path, "rb")
    data = np.load(f)
    f.close()

    return data


if __name__ == "__main__":
    '''
    Get classification performances.
    '''

    performance_path = "resources/performance_risk.np"
    val_accuracy, test_accuracy, test_precision, test_recall, test_specificity = load(
        performance_path)
    accuracies = []
    precisions = []
    recalls = []
    specificities = []

    for fold in range(len(val_accuracy[1000])):
        best_val_accuracy = 0
        best_accuracy = 0
        best_precision = 0
        best_recall = 0
        best_specificity = 0
        key_sum = 0
        key_val_sum = 0

        for key in sorted(val_accuracy):
            if (val_accuracy[key][fold] >= best_val_accuracy):
                if val_accuracy[key][fold] > best_val_accuracy:
                    best_keys = []

                best_keys.append(key)
                best_val_accuracy = val_accuracy[key][fold]
                best_accuracy = test_accuracy[key][fold]
                best_precision = test_precision[key][fold]
                best_recall = test_recall[key][fold]
                best_specificity = test_specificity[key][fold]
                best_key = key

            key_sum = key_sum + key * val_accuracy[key][fold]
            key_val_sum = key_val_sum + val_accuracy[key][fold]

        best_key = 25000
        delta_key = 9999999
        mean_key = np.mean(best_keys)

        for key in sorted(best_keys):
            delta_tmp = np.abs(key - mean_key)
            if delta_tmp <= delta_key:
                best_key = key
                delta_key = delta_tmp

        best_val_accuracy = val_accuracy[best_key][fold]
        best_accuracy = test_accuracy[best_key][fold]
        best_precision = test_precision[best_key][fold]
        best_recall = test_recall[best_key][fold]
        best_specificity = test_specificity[best_key][fold]
        accuracies.append(best_accuracy)
        precisions.append(best_precision)
        recalls.append(best_recall)
        specificities.append(best_specificity)

    print round(np.mean(np.array(accuracies) * 100.0), 1), "+-", round(np.std(np.array(accuracies) * 100.0), 1), "-", \
        round(np.mean(np.array(precisions) * 100.0), 1), "+-", round(np.std(np.array(precisions) * 100.0), 1), "-", \
        round(np.mean(np.array(recalls) * 100.0), 1), "+-", round(np.std(np.array(recalls) * 100.0), 1), "-", \
        round(np.mean(np.array(specificities) * 100.0),
              1), "+-", round(np.std(np.array(specificities) * 100.0), 1)
