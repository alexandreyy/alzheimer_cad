'''
Created on 26 de fev de 2016

@author: Alexandre Yukio Yamashita
'''

from files import Files
from nifti import Nifti
from utils import group_brains_by_patient_id, group_brains_by_category, get_patient_id, get_category
import numpy as np
import os.path


class Kfold:
    '''
    Create, save and load k-fold data.
    '''

    def __init__(self, path=None, folds=5):
        if path is not None:
            self.get_data_paths(path=path, folds=folds)
        else:
            self.fold_itens = np.array([])

    def get_data_paths(self, path, folds=5):
        '''
        Get splitted data paths.
        '''

        total_folds_itens = np.zeros(folds)

        # Get data files.
        brains_data = Files(path)
        brains_data = brains_data.get_file_names()
        total_brains = len(brains_data)
        fold_size = np.int(total_brains * 1.0 / folds + 0.5)

        # Get total of brains by category.
        brains_by_patient = group_brains_by_patient_id(brains_data)
        patient_ids = brains_by_patient.keys()
        np.random.shuffle(patient_ids)
        fold_itens = []
        for _ in range(folds):
            fold_itens.append([])

        # Split data, avoiding brains of the same patient being moved to
        # different sets.
        for patient in patient_ids:
            delta_folds_itens = fold_size - total_folds_itens
            selected_fold = np.argmax(delta_folds_itens)

            for brain in brains_by_patient[patient]:
                fold_itens[selected_fold].append(brain)
                total_folds_itens[selected_fold] += 1

        self.fold_itens = np.array(fold_itens)

    def split_train_data(self, paths, ratio=None):
        '''
        Get splitted data paths.
        '''

        if ratio == None:
            ratio = 0.05 / (1 - 1.0 / len(self.fold_itens))

        # Get data files.
        brains_data = Files("")
        brains_data.paths = paths
        brains_data = brains_data.get_file_names()
        total_brains = len(brains_data)
        validation_size = np.int(total_brains * ratio)

        # Get total of brains by category.
        brains_by_patient = group_brains_by_patient_id(brains_data)
        validation_paths = []
        train_paths = []
        brains_by_category = group_brains_by_category(brains_data)
        statistic = {}

        # Initialize statistic data.
        for label in brains_by_category:
            statistic[label] = np.round(
                len(brains_by_category[label]) * validation_size * 1.0 / total_brains)

        # Create train and validation set.
        for label in brains_by_category:
            index_brains = 0

            while statistic[label] > 0:
                patient_id = get_patient_id(
                    brains_by_category[label][index_brains])
                brains_patient = brains_by_patient[patient_id]

                for brain_patient in brains_patient:
                    validation_paths.append(brain_patient)
                    statistic[get_category(brain_patient)] -= 1

                index_brains += 1

        for brain in brains_data:
            if brain not in validation_paths:
                train_paths.append(brain)

        # Shuffle data.
        np.random.shuffle(np.array(validation_paths))
        np.random.shuffle(np.array(train_paths))

        return train_paths, validation_paths

    def save(self, path):
        '''
        Save fold itens.
        '''

        f = file(path, "wb")
        np.save(f, self.fold_itens)
        f.close()

    def load(self, path):
        '''
        Load fold itens.
        '''

        f = file(path, "rb")
        self.fold_itens = np.load(f)
        f.close()

    def show(self):
        '''
        Show data distribution in folds.
        '''

        brains_path = []
        for itens in self.fold_itens:
            for item in itens:
                brains_path.append(item)

        self.statistic = np.zeros((len(self.fold_itens), len(
            group_brains_by_category(self.fold_itens[0]))))

        index_fold = 0
        for fold in self.fold_itens:
            brains_by_category = group_brains_by_category(fold)

            index_category = 0
            for category in brains_by_category:
                total_brains_fold = len(brains_by_category[category])
                print "%s: %d (%.2f%%)" % (category, total_brains_fold, total_brains_fold * 100.0 / len(fold))
                self.statistic[index_fold, index_category] = total_brains_fold
                index_category += 1

            index_fold += 1
            print ""

        brains_by_category = group_brains_by_category(brains_path)

        print "-------------------------------"
        index_category = 0
        for category in brains_by_category:
            total_brains_fold = len(brains_by_category[category])
            print "%s: %d (%.2f%%)" % (category, total_brains_fold, total_brains_fold * 100.0 / len(brains_path))
            index_category += 1

        index_fold += 1
        print ""

        print "Total: ", len(brains_path)
        print "Patients: ", len(group_brains_by_patient_id(brains_path)), "\n"

    def get_train_test_path(self, fold_index=0):
        '''
        Get train and test path by fold.
        '''

        test_data = self.fold_itens[fold_index]
        train_data = []

        for i in range(len(self.fold_itens)):
            if fold_index != i:
                if len(train_data) == 0:
                    train_data = np.array(self.fold_itens[i])
                else:
                    train_data = np.hstack((train_data, self.fold_itens[i]))

        test_data = np.array(test_data)

        return train_data, test_data

    def get_train_test_validation_data(self, path, fold_index=0, one_hot_encode=False):
        '''
        Get train, test and validation data.
        '''

        train_path, test_path = self.get_train_test_path(fold_index)
        train_path, validation_path = self.split_train_data(train_path)
        train_data = self._get_data(
            root_path=path, file_paths=train_path, one_hot_encode=one_hot_encode)
        validation_data = self._get_data(
            root_path=path, file_paths=validation_path, one_hot_encode=one_hot_encode)
        test_data = self._get_data(
            root_path=path, file_paths=test_path, one_hot_encode=one_hot_encode)

        return train_data[0], train_data[1], test_data[0], test_data[1], validation_data[0], validation_data[1]

    def get_train_data(self, path, fold_index=0, one_hot_encode=False):
        '''
        Get train data.
        '''

        train_path, _ = self.get_train_test_path(fold_index)
        train_path, _ = self.split_train_data(train_path)
        train_data = self._get_data(
            root_path=path, file_paths=train_path, one_hot_encode=one_hot_encode)

        return train_data[0], train_data[1]

    def _get_data(self, root_path="", file_paths="", one_hot_encode=False):
        '''
        Get data from file paths.
        '''

        y = []
        X = []
        self.paths = []

        if file_paths is not None:
            file_paths = np.unique(file_paths)

            for path in file_paths:
                try:
                    if os.path.exists(root_path + path):
                        brain = Nifti(str(root_path) + str(path))
                        data = brain.data
                        data[np.where(np.isnan(data))] = 0
                        data = data.flatten()

                        if np.mean(data) != 0:
                            if "_Normal_" in path:
                                y.append(0)
                            # elif "_AD_" in path:
                            #    y.append(1)
                            # elif "_pMCI_" in path:
                            #    y.append(2)
                            else:
                                # y.append(3)
                                y.append(1)

                            X.append(data)
                            self.paths.append(path)
                        else:
                            print "Ignoring empty image: ", root_path + path
                    else:
                        print "File does not exist.", root_path + path
                except:
                    print("Error in reading " + path)

        X = np.array(X)
        y = np.array(y)

        # Encode labels, if requested.
        if one_hot_encode:
            y = self._one_hot(y, len(np.unique(y)))

        return X, y

    def _one_hot(self, x, n):
        '''
        Encode label vector in one hot format.
        '''

        if type(x) == list:
            x = np.array(x)
        x = x.flatten()
        o_h = np.zeros((len(x), n))
        o_h[np.arange(len(x)), x] = 1

        return o_h


if __name__ == "__main__":
    '''
    Create folds with low standard deviation of sizes.
    '''

    # File path format:
    # <Subject ID>       + "_" + <label>  + "_" + <Sex> + "_" + <Age> + "_" +   <Date>     + "_brain.nii.gz"
    # "002_0295_I118692" + "_" + "Normal" + "_" +  "M"  + "_" +  "85" + "_" + "2006-11-02" + "_brain.nii.gz"
    #
    # Examples:
    # 002_0295_I118692_Normal_M_85_2006-11-02_brain.nii.gz
    # 002_0619_I120964_AD_M_79_2008-08-13_brain.nii.gz
    # 002_1155_I40845_sMCI_M_57_2006-12-14_brain.nii.gz
    # 002_1268_I78668_pMCI_M_83_2007-09-21_brain.nii.gz
    #

    database_path = "../resources/dataset_pre_processed/"
    k_fold_path = "../resources/k_fold_10.bin"
    index_iteration = 0
    min_std = 999
    max_iterations = 10000

    while True:
        k_fold = Kfold(path=database_path, folds=10)
        k_fold.show()
        std = np.std(k_fold.statistic, axis=0)
        std = np.mean(std)

        if std < min_std:
            min_std = std
            k_fold.save(k_fold_path)

        print index_iteration, min_std

        index_iteration += 1

        if index_iteration >= max_iterations:
            index_iteration = 0
            break
