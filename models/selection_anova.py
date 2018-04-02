'''
Created on 26 de fev de 2016

@author: Alexandre Yukio Yamashita
'''

from sklearn.feature_selection import SelectKBest
import numpy as np
from utils import group_brains_by_patient_id, get_patient_id
from sklearn.feature_selection import f_classif


class SelectKBestAnova:
    '''
    Compute ANOVA scores and select features.
    '''

    def __init__(self, k = None, iterations = 1):
        self.k = k
        self.iterations = iterations
        self.scores_ = None


    def get_subsample_by_patient(self, X, y, patients, paths, ratio = 0.9):
        '''
        Get subset from samples.
        '''

        size = int(len(patients) * ratio)
        np.random.shuffle(patients)

        X_temp = []
        y_temp = []
        paths_temp = []
        selected = patients[:size]

        for i in range(len(X)):
            if paths[i] in selected and paths[i] not in paths_temp:
                X_temp.append(X[i])
                y_temp.append(y[i])
                paths_temp.append(paths[i])

            if len(paths_temp) == size:
                break

        X_temp = np.array(X_temp)
        y_temp = np.array(y_temp)

        return X_temp, y_temp


    def fit(self, X, y, paths, ratio = 0.9, iterations = 0):
        '''
        Compute ANOVA scores.
        '''

        if iterations != 0:
            self.iterations = iterations

        self.labels_ = np.unique(y)
        brains_by_patient = group_brains_by_patient_id(paths)
        paths = np.array([get_patient_id(paths[i]) for i in range(len(paths))])
        self.scores_ = X[0] * 0.0

        for i in range(self.iterations):
            selection = SelectKBest(f_classif, k = X.shape[1])
            randomize = range(len(X))
            np.random.shuffle(randomize)
            X = X[randomize]
            y = y[randomize]
            paths = paths[randomize]

            X_temp, y_temp = self.get_subsample_by_patient(X = X, y = y, patients = list(brains_by_patient.keys()), paths = paths, ratio = ratio)
            selection.fit(X_temp, y_temp)
            scores = selection.scores_
            scores[np.where(np.logical_or(np.isnan(scores), np.isinf(scores)))] = 0.0
            self.scores_ = np.max(np.vstack((scores, self.scores_)), axis = 0)

            del scores
            del X_temp
            del y_temp

            print i, np.mean(self.scores_)


    def transform(self, X, k = None):
        '''
        Select features using scores.
        '''

        if k is not None:
            self.k = k

        X_transform = self.select_by_feature_importance(X, self.scores_, self.k)

        return X_transform


    def save(self, path):
        '''
        Save scores.
        '''

        f = file(path, "wb")
        np.save(f, self.scores_)
        f.close()


    def load(self, path):
        '''
        Load scores.
        '''

        f = file(path, "rb")
        self.scores_ = np.load(f)
        f.close()


    def select_by_feature_importance(self, X, feature_importance, k):
        '''
        Select features with k highest scores.
        '''

        feature_importance[np.where(np.isnan(feature_importance))] = 0
        mask = np.zeros(feature_importance.shape, dtype = bool)
        mask[np.argsort(feature_importance, kind = "mergesort")[-k:]] = 1

        return X[:, np.where(np.equal(mask, 1))][:, 0, :]
