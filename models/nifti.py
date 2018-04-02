'''
Created on 10 de set de 2015

@author: Alexandre Yukio Yamashita
'''
from image import Image
import nibabel as nib
import numpy as np


class Nifti:
    '''
    Read and write Nifti file.
    '''

    path = ""


    def __init__(self, path = "", data = None, affine = None):
        if path != "":
            self.load(path)

        if data is not None and affine is not None:
            self.nifti_data = nib.Nifti1Image(data, affine)

        if data is not None and affine is None:
            path_parameters = 'resources/brain_parameters.bin'
            f = file(path_parameters, "rb")
            affine = np.load(f)
            f.close()

            self.nifti_data = nib.Nifti1Image(data, affine)


    @property
    def data(self):
        '''
        Get data from Nifti data.
        '''

        return self.nifti_data.get_data()


    def set_data(self, data):
        '''
        Set brain data.
        '''

        self.nifti_data = nib.Nifti1Image(data, self.nifti_data.affine)


    def image_by_axis(self, axis, depth = None):
        '''
        Get image by axis.
        '''

        if depth is None:
            depth = self.data.shape[axis] / 2

        if axis == 0:
            data = self.data[depth, :, :]
        elif axis == 1:
            data = self.data[:, depth, :]
        else:
            data = self.data[:, :, depth]

        return data


    def load(self, path):
        '''
        Load data from path.
        '''

        self.path = path
        self.nifti_data = nib.load(path)


    def save(self, path):
        '''
        Save Nifti file.
        '''

        nib.save(self.nifti_data, path)


    def plot(self, axis = 1, depth = None):
        '''
        Plot image.
        '''

        data = self.image_by_axis(axis, depth)
        image = Image(data = data)
        image.plot()


if __name__ == "__main__":
    '''
    Load and plot Nifti.
    '''
    path = '../resources/mean_brain_rm.nii.gz'

    # Load and plot brain image.
    nifti = Nifti(path)
    nifti.plot(0)
    nifti.plot(1)
    nifti.plot(2)
