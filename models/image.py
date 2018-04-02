# -*- coding: utf-8 -*-
'''
Created on Jul 23, 2015

@author: Alexandre Yukio Yamashita
'''
import matplotlib.pyplot as plt
import numpy as np


class Image:
    '''
    Read, save and plot image.
    '''

    data = None
    path = None


    def __init__(self, path = None, data = None):
        self._set_image_parameters(path, data)

        if self.path is not None:
            self.load(self.path)


    def _set_image_data(self, data):
        '''
        Set image data.
        '''

        # Check if image is in rgb or gray scale
        if len(data.shape) == 3:
            # Image is in rgb.
            self.height, self.width, self.channels = data.shape
        else:
            # Image is in gray scale.
            self.height, self.width = data.shape
            self.channels = 1

        self.data = data


    def _set_image_parameters(self, path = None, data = None):
        '''
        Configure image.
        '''

        if path is not None:
            self.path = path

        if data is not None:
            self._set_image_data(data)


    def _configure_plot(self):
        '''
        Configure plot to display image.
        '''

        # Remove warning for Source ID not found.
        # The warning is a issue from matplotlib.
        import warnings
        warnings.simplefilter("ignore")

        # Configure pyplot.
        frame = plt.gca()
        frame.axes.get_xaxis().set_ticklabels([])
        frame.axes.get_yaxis().set_ticklabels([])
        frame.axes.set_axis_off()

        is_gray = len(self.data.shape) < 3

        if is_gray:
            plt.imshow(self.data, cmap = plt.get_cmap("gray"))
        else:
            plt.imshow(self.data)


    def plot(self, image = None):
        '''
        Plot image.
        '''
        self._set_image_parameters(data = image)
        self._configure_plot()
        plt.show()
