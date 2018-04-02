'''
Created on 21 de fev de 2017

@author: Alexandre Yukio Yamashita
'''

import ntpath
import os.path
import glob
import itertools
from shutil import move, rmtree, copyfile
from multiprocessing import Pool, freeze_support, cpu_count
from skull_stripping import skull_strip
from spatial_normalization import spatial_normalize_ants
from intensity_normalization import normalize_min_max
from downsampling import crop, downsample


def pre_processing(input_path, output_path):
    '''
    Pre-process image.
    '''

    try:
        if not os.path.exists(output_path) and os.path.exists(input_path):
            # Get filename and extension from input path.
            basename = ntpath.basename(input_path)
            extension = os.path.splitext(basename)[1]
            input_path = os.path.abspath(input_path)

            if extension == ".gz":
                extension = ".nii.gz"

            filename = basename.replace(extension, "")
            dir_path = input_path.replace(basename, "")

            # Create temporary directory.
            temp_dir_path = dir_path + "tmp_" + filename + "/"

            if not os.path.isdir(temp_dir_path):
                os.makedirs(temp_dir_path)

            # Copy original file to temporary directory.
            copyfile(input_path, temp_dir_path + filename + extension)

            # Execute pre-processing flow.
            skull_strip(temp_dir_path + filename + extension, \
                        temp_dir_path + filename + "_brain" + extension)
            spatial_normalize_ants(temp_dir_path + filename + "_brain" + extension, \
                                   'resources/mni_icbm152_t1_tal_nlin_asym_09c_brain.nii.gz', \
                                   temp_dir_path + filename + "_brain_warped" + extension)
            normalize_min_max(temp_dir_path + filename + "_brain_warped" + extension, \
                              temp_dir_path + filename + "_brain_warped_normalized" + extension)
            crop(temp_dir_path + filename + "_brain_warped_normalized" + extension, \
                 temp_dir_path + filename + "_brain_warped_normalized_cropped" + extension, \
                 24, 169, 25, 206, 5, 158)
            downsample(temp_dir_path + filename + "_brain_warped_normalized_cropped" + extension, \
                       temp_dir_path + filename + "_brain_warped_normalized_downsampled" + extension, \
                       ratio = 0.51)

            # Move output file to output path.
            move(temp_dir_path + filename + "_brain_warped_normalized_downsampled" + extension,
                 output_path)

            # Remove temporary directory.
            if os.path.isdir(temp_dir_path):
                rmtree(temp_dir_path)
    except:
        print "Error in pre-processing image from " + input_path


def execute_pre_processing(parms):
    '''
    Execute pre-processing.
    '''

    return pre_processing(*parms)


if __name__ == "__main__":
    '''
    Pre-process images.
    '''

    freeze_support()
    input_dir = "resources/dataset/"
    output_dir = "resources/dataset_pre_processed/"
    number_of_cores = cpu_count()

    # Create output directory if it does not exist.
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get input and output paths.
    input_images = glob.glob(os.path.join(input_dir, "*.nii.gz"))
    output_images = [output_dir + ntpath.basename(path) for path in input_images]

    # Pre-process images.
    p = Pool(number_of_cores)
    p.map(execute_pre_processing, itertools.izip(input_images, output_images))
