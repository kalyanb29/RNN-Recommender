from __future__ import print_function

import numpy as np
from util import data_handling as Dataset
from util import evaluation
import argparse
from Network import *


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tshuffle', help='Shuffle sequences during training.', action='store_true')
    parser.add_argument('--extended_set',
                        help='Use extended training set (contains first half of validation and test set).',
                        action='store_true')
    parser.add_argument('-d', dest='dataset', help='Directory name of the dataset.', default='', type=str)
    parser.add_argument('--dir', help='Directory name to save model.', default='', type=str)
    parser.add_argument('--metrics', help='Metrics for validation, comma separated', default='sps', type=str)
    parser.add_argument('--progress', help='Progress intervals', default='5000', type=str)
    parser.add_argument('--mpi', help='Max progress intervals', default=np.inf, type=float)
    parser.add_argument('--max_iter', help='Max number of iterations', default=np.inf, type=float)
    parser.add_argument('--max_time', help='Max training time in seconds', default=np.inf, type=float)
    parser.add_argument('--min_iter', help='Min number of iterations before showing progress', default=50000,
                        type=float)
    parser.add_argument('-b', dest='batch_size', help='Batch size', default=16, type=int)
    parser.add_argument('-l', dest='learning_rate', help='Learning rate', default=0.1, type=float)
    parser.add_argument('--max_length', help='Maximum length of sequences during training (for RNNs)', default=30,
                        type=int)
    parser.add_argument('--temp', help='temperature parameter', default=10, type=int)
    parser.add_argument('--gamma', help='gamma', default=0.5, type=float)

    Flags, args = parser.parse_known_args()
    dataset = Dataset.DataHandler(dirname=Flags.dataset, extended_training_set=Flags.extended_set, shuffle_training=Flags.tshuffle)
    network = Network()
    network.train(dataset, save_dir=os.getcwd() + '/Models')

if __name__ == '__main__':
    torch.cuda.set_device(2)
    main()
