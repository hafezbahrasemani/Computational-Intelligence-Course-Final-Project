import random
import csv
import numpy as np


def read_file(path):
    with open(path, newline='') as file:
        data = []
        labels = []
        for line in file:
            if line.isspace():
               None
            else:
                tokens = line.split(',')
                # print(tokens)
                data.append([float(tokens[0]), float(tokens[1]), float(tokens[2].split('\r')[0])])
                # labels.append()
        # shuffle_tmp = list(zip(data, labels))
        np.random.seed(2)
        np.random.shuffle(data)
        # print(data)
        # data, labels = zip(*shuffle_tmp)
        return data
