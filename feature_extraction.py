'''
Created on 21 de fev de 2017

@author: Alexandre Yukio Yamashita
'''

from models.nifti import Nifti
from models.rcm_extractor import RcmFeatureExtractor
from multiprocessing import Pool, freeze_support, cpu_count
import glob
import itertools
import os.path
import ntpath


def extract_features(featureExtractor, filter_size, input_path, output_path):
    '''
    Extract features.
    '''

    try:
        if not os.path.exists(output_path) and os.path.exists(input_path):
            nifti = Nifti(input_path)
            rcm_data = featureExtractor.compute_feature(nifti.data, filter_size)
            nifti.set_data(rcm_data)
            nifti.save(output_path)
    except:
        print "Error in extracting feature from " + input_path


def execute_extract_features(parms):
    '''
    Execute features extraction.
    '''

    return extract_features(*parms)


if __name__ == "__main__":
    '''
    Extract features.
    '''

    freeze_support()
    input_dir = "resources/dataset_pre_processed/"
    output_dir = "resources/dataset_features/"
    number_of_cores = cpu_count()
    filter_sizes = [3, 5, 7, 9]
    output_dirs = [output_dir + "features_" + str(filter_size) + "/" for filter_size in filter_sizes]
    output_dirs.append(output_dir)

    # Create output directories.
    for directory in output_dirs[::-1]:
        if not os.path.exists(directory):
            os.mkdir(directory)
    output_dirs.remove(output_dir)

    # Get input and output paths.
    input_images = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    feature_extractor = RcmFeatureExtractor((74, 92, 78))

    for filter_size, output_dir in zip(filter_sizes, output_dirs):
        output_images = [output_dir + ntpath.basename(path) for path in input_images]

        # Extract features.
        p = Pool(number_of_cores)
        p.map(execute_extract_features, \
              itertools.izip(itertools.repeat(feature_extractor),
                             itertools.repeat(filter_size),
                             input_images, output_images))
