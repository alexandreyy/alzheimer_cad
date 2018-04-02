'''
Created on 21 de fev de 2017

@author: Alexandre Yukio Yamashita
'''

from models.nifti import Nifti
import scipy.ndimage


def crop(input_path, output_path, x_start, x_end, y_start, y_end, z_start, z_end):
    '''
    Crop image.
    '''

    try:
        nifti = Nifti(input_path)
        data = nifti.data
        data = data[x_start:x_end, y_start:y_end, z_start:z_end]
        nifti.set_data(data)
        nifti.save(output_path)
    except:
        print "Error in cropping image from " + input_path


def downsample(input_path, output_path, ratio = 0.51):
    '''
    Downsample resolution.
    '''

    try:
        nifti = Nifti(input_path)
        data = nifti.data
        data = scipy.ndimage.zoom(data, ratio, order = 5)
        nifti.set_data(data)
        nifti.save(output_path)
    except:
        print "Error in downsampling image from " + input_path


if __name__ == "__main__":
    '''
    Downsample resolution.
    '''

    input_path = 'resources/sample_brain_warped_normalized.nii.gz'
    output_path = 'resources/sample_brain_warped_normalized_downsampled.nii.gz'

    # Crop and downsample image.
    crop(input_path, output_path, 24, 169, 25, 206, 5, 158)
    downsample(output_path, output_path, ratio = 0.51)
