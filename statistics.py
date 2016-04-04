"""Module to generate statistics for all text files in a directory"""

import os
from matplotlib import pyplot as plt

"""
Statistic Functions
"""


def get_avg_file_length(dir):
    """
    Get the average file length of all files in the passed in directory.

    :param dir: <String> path to directory to operate on
    :return: <Int> Line count, <Int> File count
    """
    file_count = 0
    line_count = 0
    for file in os.listdir(dir):
        file_count += 1
        if ".DS" in file:
            continue
        with open(dir + file) as f:
            line_count += len(f.readlines())
    return line_count / file_count


def get_avg_character_count(dir):
    """
    Get the average character count of all files in the passed in directory.

    :param dir: <String> path to directory to operate on
    :return: <Int> character count, <Int> file_count
    """
    file_count = 0
    char_count = 0
    for file in os.listdir(dir):
        file_count += 1
        if ".DS" in file:
            continue
        with open(dir + file) as f:
            char_count += len(f.read())
    return char_count / file_count


def get_line_counts(dir):
    """
    Get the average line count of all files in the passed in directory.

    :param dir: <String> path to directory to operate on
    :return: <List> line counts
    """
    line_counts = []
    for file in os.listdir(dir):
        if ".DS" in file:
            continue
        if os.path.isdir(dir+file):
            path = dir+file+'/'
            for file2 in os.listdir(path):
                with open(path+file2, encoding='ISO-8859-1') as f:
                    line_counts.append(len(f.readlines()))
        else:
            with open(dir + file) as f:
                line_counts.append(len(f.readlines()))
    return line_counts


"""
Visualization Functions
"""


def make_line_counts_histogram(line_counts, bins=10):
    """
    Generates a histogram of line counts given a .

    :param line_counts: <List> line counts
    :param bins: <Int> number of bins to group into in histogram
    :return:
    """
    plt.xlabel('Lines')
    plt.ylabel('Count')
    plt.hist(line_counts, bins=bins, range=(0, 30000))
    plt.show()


"""
Main Function
"""


if __name__ == '__main__':
    direc = '/Users/nikolaivogler/manybooks_sanitized/'

    print(get_avg_file_length(direc))
    print(get_avg_character_count(direc))
    line_counts = get_line_counts(direc)
    print(line_counts)
    make_line_counts_histogram(line_counts, bins=20)