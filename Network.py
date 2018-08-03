from __future__ import print_function

import glob
import os
import random
import re
import sys
from net import *
from time import time
from util import evaluation
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from metaModule import *
from target_selection import *

class Network():
    def __init__(self,
                 max_length=30,
                 batch_size=16,
                 temperature=10,
                 gamma=0.5):
        super(Network, self).__init__()

        self.max_length = max_length
        self.batch_size = batch_size
        self.temperature = temperature
        self.gamma = gamma

        self._input_type = 'float32'
        self.target_selection = SelectTargets()

        self.name = "Network"
        self.metrics = {'recall': {'direction': 1},
                        'precision': {'direction': 1},
                        'sps': {'direction': 1},
                        'user_coverage': {'direction': 1},
                        'item_coverage': {'direction': 1},
                        'ndcg': {'direction': 1},
                        'blockbuster_share': {'direction': -1}
                        }

    def top_k_recommendations(self, sequence, k=10):
        ''' Recieves a sequence of (id, rating), and produces k recommendations (as a list of ids)
        '''

        seq_by_max_length = sequence[-min(self.max_length, len(sequence)):]  # last max length or all

        X = np.zeros((1, self.max_length, self._input_size()), dtype=self._input_type)  # input shape of the RNN
        X[0, :len(seq_by_max_length), :] = np.array(list(map(lambda x: self._get_features(x), seq_by_max_length)))

        output = self.model.predict_on_batch(X)

        # find top k according to output
        return list(np.argpartition(-output[0], list(range(k)))[:k])

    def train(self, dataset,
              max_time=np.inf,
              progress=2.0,
              autosave='All',
              save_dir='',
              min_iterations=0,
              max_iter=np.inf,
              lr = 0.01,
              load_last_model=False,
              early_stopping=None,
              validation_metrics=['sps']):
        '''Train the model based on the sequence given by the training_set

        max_time is used to set the maximumn amount of time (in seconds) that the training can last before being stop.
            By default, max_time=np.inf, which means that the training will last until the training_set runs out, or the user interrupt the program.

        progress is used to set when progress information should be printed during training. It can be either an int or a float:
            integer : print at linear intervals specified by the value of progress (i.e. : progress, 2*progress, 3*progress, ...)
            float : print at geometric intervals specified by the value of progress (i.e. : progress, progress^2, progress^3, ...)

        max_progress_interval can be used to have geometric intervals in the begining then switch to linear intervals.
            It ensures, independently of the progress parameter, that progress is shown at least every max_progress_interval.

        time_based_progress is used to choose between using number of iterations or time as a progress indicator. True means time (in seconds) is used, False means number of iterations.
        '''

        self.dataset = dataset
        self.n_items = dataset.n_items
        self.lr = lr

        if len(set(validation_metrics) & set(self.metrics.keys())) < len(validation_metrics):
            raise ValueError(
                'Incorrect validation metrics. Metrics must be chosen among: ' + ', '.join(self.metrics.keys()))

        # Load last model if needed
        iterations = 0
        epochs_offset = 0

        # Make batch generator
        # batch_generator = threaded_generator(self._gen_mini_batch(self.sequence_noise(dataset.training_set())))

        batch_generator = self._gen_mini_batch(self.dataset.training_set())

        start_time = time()
        next_save = int(progress)
        # val_costs = []
        train_costs = []
        current_train_cost = []
        epochs = []
        metrics = {name: [] for name in self.metrics.keys()}
        filename = {}
        ts_sum = 0.
        self.net = MetaLearner(None, self.n_items, self.n_items, 30, 2, self.batch_size, True)
        optimizer = optim.SGD(self.net.metalearner.parameters(), lr=self.lr, momentum=0.9, nesterov=True)
        try:
            while time() - start_time < max_time and iterations < max_iter:

                # Train with a new batch
                try:
                    batch = next(batch_generator)
                    ts = time()
                    output, target = self.prepare_networks(batch, self.batch_size)
                    cost = F.cross_entropy(output, torch.max(target, 1)[1])
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()
                    ts_sum += time() - ts
                    # print(output)
                    # print(cost)

                    # exit()

                    # if np.isnan(cost):
                    #     raise ValueError("Cost is NaN")
        #
                except StopIteration:
                    break
        #
                current_train_cost.append(cost.item())
                # current_train_cost.append(0)

                # Check if it is time to save the model
                iterations += 1

                if iterations >= next_save:
                    if iterations >= min_iterations:
                        # Save current epoch
                        epochs.append(epochs_offset + self.dataset.training_set.epochs)

                        # Average train cost
                        train_costs.append(np.mean(current_train_cost))
                        current_train_cost = []

                        # Compute validation cost
                        metrics = self._compute_validation_metrics(metrics)

                        # Print info
                        self._print_progress(iterations, epochs[-1], start_time, train_costs, metrics,
                                             validation_metrics)
                        # exit()

                        # Save model
                        run_nb = len(metrics[list(self.metrics.keys())[0]]) - 1
                        filename[run_nb] = save_dir + '/model.net'
                        self._save(filename[run_nb])

                    next_save += progress

        except KeyboardInterrupt:
            print('Training interrupted')
        #
        best_run = np.argmax(
            np.array(metrics[validation_metrics[0]]) * self.metrics[validation_metrics[0]]['direction'])
        return ({m: metrics[m][best_run] for m in self.metrics.keys()}, time() - start_time, filename[best_run])

    def _gen_mini_batch(self, sequence_generator, test=False):
        ''' Takes a sequence generator and produce a mini batch generator.
        The mini batch have a size defined by self.batch_size, and have format of the input layer of the rnn.

        test determines how the sequence is splitted between training and testing
            test == False, the sequence is split randomly
            test == True, the sequence is split in the middle

        if test == False, max_reuse_sequence determines how many time a single sequence is used in the same batch.
            with max_reuse_sequence = inf, one sequence will be used to make the whole batch (if the sequence is long enough)
            with max_reuse_sequence = 1, each sequence is used only once in the batch
        N.B. if test == True, max_reuse_sequence = 1 is used anyway
        '''

        while True:
            j = 0
            sequences = []
            batch_size = self.batch_size
            if test:
                batch_size = 1
            while j < batch_size:  # j : user order

                sequence, user_id = next(sequence_generator)

                # finds the lengths of the different subsequences
                if not test:  # training set
                    seq_lengths = sorted(
                        random.sample(range(2, len(sequence)),  # range
                                      min([self.batch_size - j, len(sequence) - 2]))  # population
                    )
                else:  # validating set
                    seq_lengths = [int(len(sequence) / 2)]  # half of len

                skipped_seq = 0
                for l in seq_lengths:
                    # target is only for rnn with hinge, logit and logsig.
                    target = self.target_selection(sequence[l:], test=test)
                    if len(target) == 0:
                        skipped_seq += 1
                        continue
                    start = max(0, l - self.max_length)  # sequences cannot be longer than self.max_length
                    # print(target)
                    sequences.append([user_id, sequence[start:l], target])
                # print([user_id, sequence[start:l], target])

                j += len(seq_lengths) - skipped_seq  # ?????????

            if test:
                yield self._prepare_input(sequences), [i[0] for i in sequence[seq_lengths[0]:]]
            else:
                yield self._prepare_input(sequences)

    def _print_progress(self, iterations, epochs, start_time, train_costs, metrics, validation_metrics):
        '''Print learning progress in terminal
        '''
        print(self.name, iterations, "batchs, ", epochs, " epochs in", time() - start_time, "s")
        print("Last train cost : ", train_costs[-1])
        for m in self.metrics:
            print(m, ': ', metrics[m][-1])
            if m in validation_metrics:
                print('Best ', m, ': ',
                      max(np.array(metrics[m]) * self.metrics[m]['direction']) * self.metrics[m]['direction'])

        print('-----------------')

        # Print on stderr for easier recording of progress
        print(iterations, epochs, time() - start_time, train_costs[-1],
              ' '.join(map(str, [metrics[m][-1] for m in self.metrics])), file=sys.stderr)

    def prepare_networks(self, batch, batch_size, flag = 'Train'):
        if flag == 'Train':
            self.net.metalearner.train()
        else:
            self.net.metalearner.eval()
        with torch.no_grad():
            inputs = Variable(torch.Tensor(len(batch[0]), batch_size, self.n_items)).cuda()
            for i in range(len(batch[0])):
                for j in range(batch_size):
                    inputs[i][j] = Variable(torch.Tensor(batch[0][i][j]))
            target = Variable(torch.Tensor(batch_size, self.n_items)).cuda()
            for i in range(batch_size):
                target[i] = Variable(torch.Tensor(batch[1][i]))
        output = self.net.takeAction(inputs, batch_size)
        return output, target# (B, max_length, hidden)

    def _prepare_input(self, sequences):
        """ Sequences is a list of [user_id, input_sequence, targets]
        """
        # print("_prepare_input()")
        batch_size = len(sequences)

        # Shape of return variables
        X = np.zeros((self.max_length, batch_size, self.n_items), dtype=self._input_type)  # input of the RNN
        length = np.zeros((batch_size), dtype=np.int32)
        Y = np.zeros((batch_size, self.n_items), dtype=np.float32)  # output target

        for i, sequence in enumerate(sequences):
            user_id, in_seq, target = sequence
            seq_features = np.array(list(map(lambda x: self._get_features(x), in_seq)))
            X[:len(in_seq), i, :] = seq_features  # Copy sequences into X

            length[i] = len(in_seq)
            Y[i][target[0][0]] = 1.

        return X, Y, length

    def _compute_validation_metrics(self, metrics):
        """
        add value to lists in metrics dictionary
        """
        ev = evaluation.Evaluator(self.dataset, k=10)
        for batch_input, goal in self._gen_mini_batch(self.dataset.validation_set(epochs=1), test=True):
            output, _ = self.prepare_networks(batch_input, 1, 'Test')
            predictions = np.argpartition(-output.detach().cpu().numpy(), list(range(10)), axis=-1)[0, :10]
            # print("predictions")
            # print(predictions)
            ev.add_instance(goal, predictions)

        metrics['recall'].append(ev.average_recall())
        metrics['sps'].append(ev.sps())
        metrics['precision'].append(ev.average_precision())
        metrics['ndcg'].append(ev.average_ndcg())
        metrics['user_coverage'].append(ev.user_coverage())
        metrics['item_coverage'].append(ev.item_coverage())
        metrics['blockbuster_share'].append(ev.blockbuster_share())

        # del ev
        ev.instances = []

        return metrics

    def _get_features(self, item):
        '''Change a tuple (item_id, rating) into a list of features to feed into the RNN
        features have the following structure: [one_hot_encoding]
        '''

        # item = (item_id, rating)

        one_hot_encoding = np.zeros(self.n_items)
        one_hot_encoding[item[0]] = 1
        return one_hot_encoding

    def _save(self, filename):
        '''Save the parameters of a network into a file
        '''
        print('Save model in ' + filename)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        torch.save(self.net, filename)