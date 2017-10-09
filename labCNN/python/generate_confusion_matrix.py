#!/usr/bin/env python
"""
Plot a confusion matrix
"""
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import scikitplot.plotters as skplt

from argparse import ArgumentDefaultsHelpFormatter
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix

def parse_args():
    """Parse input arguments
    """
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--testlog',
                        help=('Test log file that contains the output to generate the matrix'))

    parser.add_argument('--configname',
                        help=('Configuration name'))

    parser.add_argument('--outputdir',
                        help=('Where the confusion matrix will be generated'))

    parser.add_argument('--filename',
                        help=('Confusion matrix file name'))

    args = parser.parse_args()
    return args

def read_file(path):
    real = []
    predicted = []
    with open(path) as f:
        for line in f:
            if line[0] != 'A':
                r, p = line.strip().split(' ')
                real.append(r)
                predicted.append(p)
    return real, predicted

def main():
    args = parse_args()

    real_labels, predicted_labels = read_file(args.testlog)

    skplt.plot_confusion_matrix(
        real_labels, predicted_labels,
        normalize=True,
        title=' ',
        text_fontsize="large"
    )
    plt.savefig("{}/{}/confusion_matrix.pdf".format(args.outputdir, args.configname), bbox_inches='tight')

    cm = confusion_matrix(real_labels, predicted_labels)
    np.set_printoptions(precision=2)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    diagonal = np.squeeze(np.asarray(np.matrix(cm).diagonal()))

    with open("{}/{}".format(args.outputdir, "confusion_matrix_all"),"a+") as f:
        f.write(args.configname + "|")
        for x in diagonal:
            f.write(str(x) + " ")
        f.write("\n")

if __name__ == '__main__':
    main()
