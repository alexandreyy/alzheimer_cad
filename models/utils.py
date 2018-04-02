'''
Created on 10 de set de 2015

@author: Alexandre Yukio Yamashita
'''

import numpy as np


def get_strings_with_substring(strings, substring):
    '''
    Return strings that contains substring.
    '''

    result = []

    if type(substring) == list:
        for substring_item in substring:
            for string in strings:
                if (substring_item in string) and (string not in result):
                    result.append(string)
                    break
    else:
        for string in strings:
            if (substring in string) and (string not in result):
                result.append(string)

    return result


def group_brains_by_patient_id(brains_data):
    '''
    Group brains by patient id.
    '''

    patient_ids = [get_patient_id(file_name) for file_name in brains_data]

    # Create dict to group brains.
    result = dict()
    for patient_id in np.unique(patient_ids):
        result[patient_id] = []

    for brain_index in range(len(brains_data)):
        result[patient_ids[brain_index]].append(brains_data[brain_index])

    return result


def group_brains_by_category(brains_data):
    '''
    Group brains by category.
    '''

    categories = []

    for file_name in brains_data:
        category = get_category(file_name)
        categories.append(category)

    # Create dict to group brains.
    result = dict()
    labels = ["Normal", "AD", "sMCI", "pMCI"]
    for category in labels:
        result[category] = []

    for brain_index in range(len(brains_data)):
        result[categories[brain_index]].append(brains_data[brain_index])

    return result


def get_patient_id(file_name):
    '''
    Get patient id from file name.
    '''

    splitted_text = file_name.split("_")
    result = splitted_text[0] + "_" + splitted_text[1]

    return result


def get_category(file_name):
    '''
    Get category from file name.
    '''

    splitted_text = file_name.split("_")
    result = splitted_text[3]

    return result
