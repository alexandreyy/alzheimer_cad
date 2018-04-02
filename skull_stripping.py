'''
Created on 21 de fev de 2017

@author: Alexandre Yukio Yamashita
'''

import ntpath
import os.path
from shutil import move
import os


def skull_strip(input_path, output_path):
    '''
    Extract brain removing skull and muscular tissues.
    '''

    try:
        # Get filename and extension from input path.
        basename = ntpath.basename(input_path)
        extension = os.path.splitext(basename)[1]
        input_path = os.path.abspath(input_path)

        if extension == ".gz":
            extension = ".nii.gz"

        filename = basename.replace(extension, "")
        dir_path = input_path.replace(basename, "")

        # Extract brain.
        command_txt = os.getenv('FSLDIR').rstrip() + "/bin/bet " + \
                        dir_path + filename + " " + \
                        dir_path + filename + "_brain" + \
                        " -R -f 0.5 -g 0"

        print command_txt
        os.system(command_txt)
        move(dir_path + "/" + filename + "_brain" + extension, output_path)
    except:
        print "Error in extracting brain from " + input_path


if __name__ == "__main__":
    '''
    Skull stripping.
    '''

    input_path = 'resources/sample.nii.gz'
    output_path = ''

    # Skull strip image.
    skull_strip(input_path, "resources/sample_brain.nii.gz")
