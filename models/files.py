'''
Created on 05/04/2015

@author: Alexandre Yukio Yamashita
'''

from glob import glob
from os.path import basename


class Files:
    '''
    List of file paths.
    '''

    def __init__(self, path):
        self.paths = glob(path + '*.nii.gz')


    def get_file_names(self):
        '''
        Get relative file paths from directory path.
        '''

        file_names = [basename(path) for path in self.paths]

        return file_names
