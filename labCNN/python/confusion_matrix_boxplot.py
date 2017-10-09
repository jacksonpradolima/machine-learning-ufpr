#!/usr/bin/env python
"""
Plot confusion matrix boxplot
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

from argparse import ArgumentDefaultsHelpFormatter
from argparse import ArgumentParser

def parse_args():
    """Parse input arguments
    """
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--cmall', help=('Confusion matrix all'))

    parser.add_argument('--outputdir', help=('Where the boxplot will be generated'))

    args = parser.parse_args()
    return args

def read_file(path):
    matrix_dist = []
    names = []

    with open(path) as f:
        for line in f:
            name, diagonal = line.strip().split('|')
            names.append(name)
            matrix_dist.append([float(x) for x in diagonal.strip().split(' ')])

    return names, matrix_dist

def main():
    args = parse_args()

    names, matrix_dist = read_file(args.cmall)

    plt.figure()
    plt.title("")
    plt.boxplot(matrix_dist, labels=names)
    plt.grid(True)
    plt.ylabel('Escore')
    plt.tick_params(labelsize="medium")
    plt.tight_layout()
    plt.savefig(args.outputdir + "distribution_confusion_matrix.pdf")

if __name__ == '__main__':
    main()
