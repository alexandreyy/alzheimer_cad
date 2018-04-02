'''
Created on 21 de fev de 2017

@author: Alexandre Yukio Yamashita
'''

from nifti import Nifti
import numpy as np


class RcmFeatureExtractor:
    '''
    Residual center of mass extractor.
    '''

    indices = None

    def __init__(self, shape = None):
        if shape is not None:
            self.indices = np.indices(shape)


    def _expand_image(self, input_image, size):
        '''
        Expand image, copying data from borders.
        '''

        output_image = np.copy(input_image)

        for axis in range(len(input_image.shape)):
            subimage_position = list(output_image.shape)
            subimage_position[axis] = 1
            subimage = np.repeat(output_image[0:subimage_position[0], \
                                              0:subimage_position[1], \
                                              0:subimage_position[2]], \
                                size, axis = axis)
            output_image = np.concatenate((subimage, output_image), axis = axis)

            subimage_position = len(output_image.shape) * [0]
            subimage_position[axis] = output_image.shape[axis] - 1
            subimage = np.repeat(output_image[subimage_position[0]:output_image.shape[0], \
                                              subimage_position[1]:output_image.shape[1], \
                                              subimage_position[2]:output_image.shape[2]], \
                                size, axis = axis)
            output_image = np.concatenate((output_image, subimage), axis = axis)

        return output_image


    def _reduce_image(self, input_image, size):
        '''
        Reduce image, removing data from borders.
        '''

        output_image = np.copy(input_image[size:(input_image.shape[0] - size), \
                                           size:(input_image.shape[1] - size), \
                                           size:(input_image.shape[2] - size)])

        return output_image


    def _compute_center_of_mass(self, input_image, filter_size, filter_center, stride, invert_image):
        '''
        Compute center of mass.
        '''

        # Expand dimensions to compute position * image.
        output_image = np.copy(input_image)
        image_expanded = np.expand_dims(input_image, axis = 0)
        density_positions = np.multiply(image_expanded, self.indices)

        # Compute area tables.
        sum_density_positions = np.copy(density_positions)
        sum_density = np.copy(input_image)

        for i in range(len(density_positions)):
            sum_density_positions = sum_density_positions.cumsum(axis = i + 1)
            sum_density = sum_density.cumsum(axis = i)

        # Include zeros in first positions of area tables.
        for i in range(len(density_positions)):
            shape = list(sum_density.shape)
            shape[i] = 1
            sum_density = np.concatenate((np.zeros(shape), sum_density), axis = i)

            shape = list(sum_density_positions.shape)
            shape[i + 1] = 1
            sum_density_positions = np.concatenate((np.zeros(shape), sum_density_positions), axis = i + 1)

        # Iterate over all image positions.
        image_limit = np.subtract(input_image.shape, filter_size)
        axis_0 = 0
        filter_sum_density = []
        filter_sum_density_positions = []
        list_positions = []

        while axis_0 < image_limit[0]:
            axis_1 = 0
            end_axis_0 = axis_0 + filter_size

            while axis_1 < image_limit[1]:
                axis_2 = 0
                end_axis_1 = axis_1 + filter_size

                while axis_2 < image_limit[2]:
                    end_axis_2 = axis_2 + filter_size

                    filter_sum_density.append((sum_density[end_axis_0, axis_1, axis_2], sum_density[end_axis_0, end_axis_1, end_axis_2], \
                        sum_density[axis_0, end_axis_1, axis_2], sum_density[axis_0, axis_1, end_axis_2], \
                        - sum_density[end_axis_0, end_axis_1, axis_2], -sum_density[axis_0, end_axis_1, end_axis_2], \
                        - sum_density[end_axis_0, axis_1, end_axis_2], -sum_density[axis_0, axis_1, axis_2]))

                    filter_sum_density_positions.append((sum_density_positions[:, end_axis_0, axis_1, axis_2], sum_density_positions[:, end_axis_0, end_axis_1, end_axis_2], \
                        sum_density_positions[:, axis_0, end_axis_1, axis_2], sum_density_positions[:, axis_0, axis_1, end_axis_2], \
                        - sum_density_positions[:, end_axis_0, end_axis_1, axis_2], -sum_density_positions[:, axis_0, end_axis_1, end_axis_2], \
                        - sum_density_positions[:, end_axis_0, axis_1, end_axis_2], -sum_density_positions[:, axis_0, axis_1, axis_2]))

                    list_positions.append((axis_0 + filter_center, axis_1 + filter_center, axis_2 + filter_center))

                    axis_2 += stride

                axis_1 += stride

            axis_0 += stride

        # Compute sum of densities in window.
        filter_sum_density = np.expand_dims(np.sum(filter_sum_density, axis = 1), axis = 1)
        filter_sum_density_positions = np.sum(np.array(filter_sum_density_positions), axis = 1)

        # Compute centroid positions.
        centroid_positions = filter_sum_density_positions * 1.0 / filter_sum_density + 0.5
        centroid_positions = centroid_positions.astype(np.int)

        # Adjust data to avoid overflow.
        for axis in range(len(density_positions)):
            centroid_positions[:, axis] = np.clip(centroid_positions[:, axis], 0, input_image.shape[axis] - 1)

        # Construct output image.
        for position, centroid in zip(list_positions, centroid_positions):
            output_image[position] = input_image[centroid[0], centroid[1], centroid[2]]

        return output_image


    def _compute_mean(self, input_image, filter_size, filter_center, stride):
        '''
        Compute mean.
        '''

        # Expand dimensions.
        output_image = np.copy(input_image)
        sum_density = np.copy(input_image)

        for i in range(len(np.shape(output_image))):
            sum_density = sum_density.cumsum(axis = i)

        # Include zeros in first positions of area tables.
        for i in range(len(np.shape(output_image))):
            shape = list(sum_density.shape)
            shape[i] = 1
            sum_density = np.concatenate((np.zeros(shape), sum_density), axis = i)

        # Iterate over all image positions.
        image_limit = np.subtract(input_image.shape, filter_size)
        axis_0 = 0
        filter_sum_density = []
        list_positions = []

        while axis_0 < image_limit[0]:
            axis_1 = 0
            end_axis_0 = axis_0 + filter_size

            while axis_1 < image_limit[1]:
                axis_2 = 0
                end_axis_1 = axis_1 + filter_size

                while axis_2 < image_limit[2]:
                    end_axis_2 = axis_2 + filter_size

                    filter_sum_density.append((sum_density[end_axis_0, axis_1, axis_2], sum_density[end_axis_0, end_axis_1, end_axis_2], \
                        sum_density[axis_0, end_axis_1, axis_2], sum_density[axis_0, axis_1, end_axis_2], \
                        - sum_density[end_axis_0, end_axis_1, axis_2], -sum_density[axis_0, end_axis_1, end_axis_2], \
                        - sum_density[end_axis_0, axis_1, end_axis_2], -sum_density[axis_0, axis_1, axis_2]))

                    list_positions.append((axis_0 + filter_center, axis_1 + filter_center, axis_2 + filter_center))

                    axis_2 += stride

                axis_1 += stride

            axis_0 += stride

        # Compute sum of densities in window.
        filter_sum_density = np.expand_dims(np.sum(filter_sum_density, axis = 1), axis = 1)

        # Construct output image.
        for position, index in zip(list_positions, range(len(list_positions))):
            output_image[position] = filter_sum_density[index] / (filter_size ** 3)

        return output_image


    def compute_feature(self, input_image, filter_size = 5, stride = 1, expand_image = True, invert_image = True):
        '''
        Compute RCM descriptor.
        '''

        pre_processed_image = np.copy(input_image)
        filter_center = filter_size / 2

        if self.indices is None or expand_image:
            # Expand image to compute lloyd features in all positions of image.
            if expand_image:
                pre_processed_image = self._expand_image(pre_processed_image, filter_center)

            # Initialize indices.
            self.indices = np.indices(pre_processed_image.shape)

        # Invert image.
        if invert_image:
            max_image = np.max(pre_processed_image)
            pre_processed_image = max_image - pre_processed_image

        # Add one and subtract minimum value to avoid vanishment of positions.
        min_image = np.min(pre_processed_image)
        pre_processed_image = pre_processed_image - min_image + 1

        # Create output image.
        output_image = np.copy(pre_processed_image)

        # Compute center of mass.
        image_center_of_mass = self._compute_center_of_mass(output_image, filter_size, filter_center, stride, invert_image)

        # Convert image to original range of values.
        image_center_of_mass = image_center_of_mass + min_image - 1

        # Invert image.
        if invert_image:
            image_center_of_mass = max_image - image_center_of_mass

        # Blur image with mean filter.
        image_center_of_mass_smoothed = self._compute_mean(image_center_of_mass, filter_size, filter_center, stride)

        # Reduce image to original size.
        if expand_image:
            image_center_of_mass_smoothed = self._reduce_image(image_center_of_mass_smoothed, filter_center)

        # Compute RCM.
        output_image = input_image - image_center_of_mass_smoothed

        return output_image


if __name__ == "__main__":
    '''
    Extract features.
    '''

    input_path = "../resources/sample_pre_processed.nii.gz"
    output_path = "../resources/sample_rcm.nii.gz"
    rcm_feature_extractor = RcmFeatureExtractor((74, 92, 78))

    # Extract RCM descriptor.
    nifti = Nifti(input_path)
    rcm_data = rcm_feature_extractor.compute_feature(nifti.data)
    nifti.set_data(rcm_data)
    nifti.save(output_path)
