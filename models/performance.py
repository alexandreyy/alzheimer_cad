'''
Created on 16/11/2015

@author: Alexandre Yukio Yamashita
'''

from sklearn import metrics
from sklearn.metrics import confusion_matrix as c_matrix
import math


def compute_confusion_matrix(true_labels, predicted_labels):
    '''
    Print and get confusion matrix.
    '''
    result = c_matrix(true_labels, predicted_labels)

    # Show final results.
    print "\nConfusion matrix:"
    print "(real , predicted) | count"
    print "(%s , %s) | %f" % ("False", "False", result[0, 0] * 1.0 / (result[0, 0] + result[0, 1]))
    print "(%s , %s) | %f" % ("False", "True", result[0, 1] * 1.0 / (result[0, 0] + result[0, 1]))
    print "(%s , %s) | %f" % ("True", "False", result[1, 0] * 1.0 / (result[1, 0] + result[1, 1]))
    print "(%s , %s) | %f" % ("True", "True", result[1, 1] * 1.0 / (result[1, 0] + result[1, 1]))

    return result


def compute_performance_metrics(confusion_matrix):
    '''
    Compute performance metrics.
    '''
    accuracy = (confusion_matrix[1, 1] + confusion_matrix[0, 0]) * 1.0 / (confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0] + confusion_matrix[1, 1])
    precision = confusion_matrix[1, 1] * 1.0 / (confusion_matrix[0, 1] + confusion_matrix[1, 1])
    recall = confusion_matrix[1, 1] * 1.0 / (confusion_matrix[1, 0] + confusion_matrix[1, 1])
    specificity = confusion_matrix[0, 0] * 1.0 / (confusion_matrix[0, 1] + confusion_matrix[0, 0])

    if math.isnan(precision):
        precision = 0
    if math.isnan(recall):
        recall = 0
    if math.isnan(accuracy):
        accuracy = 0
    if math.isnan(specificity):
        specificity = 0

    print "Accuracy | %f" % accuracy
    print "Precision | %f" % precision
    print "Recall | %f" % recall
    print "Specificity | %f" % specificity

    return accuracy, precision, recall, specificity

