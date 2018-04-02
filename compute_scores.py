'''
Created on 26 de fev de 2016

@author: Alexandre Yukio Yamashita
'''

from models.selection_anova import SelectKBestAnova
import os
import numpy as np
from models.k_fold import Kfold


if __name__ == "__main__":
    '''
    Train feature selector.
    '''

    database_path = "resources/dataset_features/"
    # k_fold_path = "resources/k_fold_10.bin"
    k_fold_path = "resources/k_fold_10_risk.bin"
    filter_sizes = [3, 5, 7, 9]
    features_paths = []
    save_path = "scores_risk"
    k_fold = Kfold()
    k_fold.load(k_fold_path)

    for filter_size in filter_sizes:
        features_paths.append(
            database_path + "features_" + str(filter_size) + "_1/")

    # For all folds, load data and train selector.
    for path in features_paths:
        k_fold = Kfold()
        k_fold.load(k_fold_path)

        for fold_index in range(len(k_fold.fold_itens)):
            # Load data from paths and train selector.
            if not os.path.exists(path + save_path + "/fold_" + str(fold_index) + "_risk.np"):
                if 'k_fold' not in locals():
                    k_fold = Kfold()
                    k_fold.load(k_fold_path)

                trX_load, y_train = k_fold.get_train_data(
                    path, fold_index=fold_index)
                paths = k_fold.paths
                del k_fold

                # Get images from AD and Normal subjects.
                # trX_load = trX_load[np.where(np.not_equal(y_train, 2))]
                # y_train = y_train[np.where(np.not_equal(y_train, 2))]
                # trX_load = trX_load[np.where(np.not_equal(y_train, 3))]
                # y_train = y_train[np.where(np.not_equal(y_train, 3))]

                # Compute scores.
                selection = SelectKBestAnova()
                selection.fit(trX_load, y_train, paths,
                              ratio=0.95, iterations=100)
                selection.save(path + save_path + "/fold_" +
                               str(fold_index) + "_risk.np")
                del selection
