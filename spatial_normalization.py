'''
Created on 21 de fev de 2017

@author: Alexandre Yukio Yamashita
'''

import ntpath
import os.path
import os
import shutil


def spatial_normalize_ants(input_path, template_path, output_path):
    '''
    Spatial normalize brain using ANTs.
    '''

    try:
        # Get filename and extension from input path.
        basename = ntpath.basename(input_path)
        extension = os.path.splitext(basename)[1]
        input_path = os.path.abspath(input_path)
        input_dir_path = input_path.replace(basename, "")

        if extension == ".gz":
            extension = ".nii.gz"

        filename = basename.replace(extension, "")
        dir_path = input_path.replace(basename, "")

        # Normalize brain.
	ants_path = "/opt/ANTS/bin"
        command_txt = "cd " + input_dir_path + " && " + \
            ants_path.replace("/bin", "/Scripts") + "/antsIntroduction.sh -d 3 -r " + \
            os.path.abspath(template_path) + " " + \
            "-i " + \
            dir_path + basename + " " + \
            "-o " + \
            dir_path + filename + "_norm_"

        print command_txt
        os.system(command_txt)

        # Remove warp data.
        template_basename = ntpath.basename(template_path)
        template_extension = os.path.splitext(template_basename)[1]

        if template_extension == ".gz":
            template_extension = ".nii.gz"

        remove_path_list = [dir_path + filename + "_norm_repaired" + extension,
                            dir_path + filename + "_norm_InverseWarp" + extension,
                            dir_path + filename + "_norm_Warp" + extension,
                            dir_path + filename + "_norm_Affine.txt",
                            dir_path + filename + ".cfg",
                            template_path.replace(template_extension, "_InverseWarp" + template_extension)]

        for remove_path in remove_path_list:
            if os.path.isfile(remove_path):
                os.remove(remove_path)

        # Rename output file.
        temp_output_path = dir_path + filename + "_norm_deformed" + extension

        if os.path.isfile(temp_output_path):
            shutil.move(temp_output_path, output_path)
    except:
        print "Error in normalizing brain from " + input_path


if __name__ == "__main__":
    '''
    Spatial normalize brain.
    '''

    input_path = 'resources/sample_brain.nii.gz'
    output_path = 'resources/sample_brain_warped.nii.gz'
    template_path = 'resources/mni_icbm152_t1_tal_nlin_asym_09c_brain.nii.gz'

    from datetime import datetime
    t1 = datetime.now()

    # Spatial normalize image.
    spatial_normalize_ants(input_path, template_path, output_path)

    t2 = datetime.now()
    delta = t2 - t1
    print delta.total_seconds()

