'''
Created on 21 de fev de 2017

@author: Alexandre Yukio Yamashita
'''

import numpy as np
from models.nifti import Nifti


def normalize_min_max(input_path, output_path):
    '''
    Normalize brain intensity with min-max.
    '''

    try:
        nifti = Nifti(input_path)
        data = nifti.data
        min_image = np.min(data)
        max_image = np.max(data)

        if max_image == min_image:
            max_image = 1.0

        data = np.divide(np.subtract(data, min_image) * 1.0, (max_image - min_image))
        nifti.set_data(data)
        nifti.save(output_path)
    except:
        print "Error in normalizing brain from " + input_path


if __name__ == "__main__":
    '''
    Normalize brain intensity.
    '''

    input_path = 'resources/sample_brain_warped.nii.gz'
    output_path = 'resources/sample_brain_warped_normalized.nii.gz'

    # Normalize brain intensity.
    normalize_min_max(input_path, output_path)
