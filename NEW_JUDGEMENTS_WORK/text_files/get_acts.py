# import the required libraries
import json

import numpy as np
import re


def return_list_of_acts(file):
    """
    input: a parameter which contains the text filename
    output: a list containing acts in abbreviated form
    """

    # maintains two list one for with dot and one for without dot
    acts_with_dot = []
    acts_without_dot = []

    result = dict()

    # check if file is a text file
    if file.endswith(".txt"):
        try:
            f = open(file, "r", encoding="utf-8")
            text = f.read()

            # regex1 looks for a letter followed by a dot one or more times followed by the word Act
            regex1 = re.compile(r'([A-Z]\.' '+)+ Act')

            # regex1 looks for a letter followed by a space one or more times followed by the word Act
            regex2 = re.compile(r'([A-Z]' '+ Act)')

            res1 = regex1.search(text)
            res2 = regex2.search(text)

            # if the result is not empty then append them to the list
            if res1:
                acts_with_dot.append(res1.group())

            if res2:
                acts_without_dot.append(res2.group())

            # remove the dots and spaces from the act
            temp = []
            for act in acts_with_dot:
                x = re.sub(r'\.', '', act)
                temp.append(x)

            temp = np.unique(temp)

            acts = [i for i in temp]
            for i in acts_without_dot:
                acts.append(i)

            # remove duplicated
            acts = np.unique(acts)

            dict_map = open('acts_dictionary.json')
            dict_map = json.load(dict_map)

            for act in acts:
                result[act] = dict_map[act]
            return result

        except IOError:
            print("There was an error while opening the file!!")
            return
