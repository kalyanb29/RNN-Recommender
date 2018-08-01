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
        try:
            while time() - start_time < max_time and iterations < max_iter:

                # Train with a new batch
                try:
                    batch = next(batch_generator)
                    ts = time()
        #             cost = self.train_function(*batch)
        #             ts_sum += time() - ts
        #             # print(output)
        #             # print(cost)
        #
        #             # exit()
        #
        #             if np.isnan(cost):
        #                 raise ValueError("Cost is NaN")
        #
                except StopIteration:
                    break
        #
        #         current_train_cost.append(cost)
        #         # current_train_cost.append(0)
        #
        #         # Check if it is time to save the model
        #         iterations += 1
        #
        #         if iterations >= next_save:
        #             if iterations >= min_iterations:
        #                 # Save current epoch
        #                 epochs.append(epochs_offset + self.dataset.training_set.epochs)
        #
        #                 # Average train cost
        #                 train_costs.append(np.mean(current_train_cost))
        #                 current_train_cost = []
        #
        #                 # Compute validation cost
        #                 metrics = self._compute_validation_metrics(metrics)
        #
        #                 # Print info
        #                 self._print_progress(iterations, epochs[-1], start_time, train_costs, metrics,
        #                                      validation_metrics)
        #                 # exit()
        #
        #                 # Save model
        #                 run_nb = len(metrics[list(self.metrics.keys())[0]]) - 1
        #                 filename[run_nb] = save_dir + self.framework + "/" + self._get_model_filename(
        #                         round(epochs[-1], 3))
        #                 self._save(filename[run_nb])
        #
        #             next_save += progress
        #
        except KeyboardInterrupt:
            print('Training interrupted')
        #
        # best_run = np.argmax(
        #     np.array(metrics[validation_metrics[0]]) * self.metrics[validation_metrics[0]]['direction'])
        # return ({m: metrics[m][best_run] for m in self.metrics.keys()}, time() - start_time, filename[best_run])

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

    def prepare_networks(self, n_items):

        self.n_items = n_items
        self.net = MetaLearner(None, self.n_items, self.n_items, 30, 2, self.batch_size, True)
        if self.recurrent_layer.embedding_size > 0:
            self.X = tf.placeholder(tf.int32, [None, self.max_length])
            word_embeddings = tf.get_variable("word_embeddings", [self.n_items, self.recurrent_layer.embedding_size])
            rnn_input = tf.nn.embedding_lookup(word_embeddings, self.X)
        # print(rnn_input) # (?, max_length, embedding_size)
        else:
            self.X = tf.placeholder(tf.float32, [None, self.max_length, self.n_items])
            rnn_input = self.X
        self.Y = tf.placeholder(tf.float32, [None, self.n_items])
        self.length = tf.placeholder(tf.int32, [None, ])

        # self.length = self.get_length(rnn_input)
        self.rnn_out, _state = self.recurrent_layer(rnn_input, self.length,
                                                    activate_f=activation)  # (B, max_length, hidden)
        self.rnn_out, _state = LSTM()
        self.output = tf.layers.dense(self.last_hidden, self.n_items, activation=None)
        # applying attention makes last_hidden have undefined rank. need to reshape

        self.softmax = tf.nn.softmax(self.output)
        self.xent = -tf.reduce_sum(self.Y * tf.log(self.softmax))
        self.cost = tf.reduce_mean(self.xent)
        # tf.summary.histogram("cost", self.cost)

        optimizer = Adam(model.parameters(), )
        self.training = optimizer.minimize(self.cost)

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
        if not self.iter:
            for batch_input, goal in self._gen_mini_batch(self.dataset.validation_set(epochs=1), test=True):
                output = self.sess.run(self.softmax, feed_dict={self.X: batch_input[0], self.length: batch_input[2]})
                predictions = np.argpartition(-output, list(range(10)), axis=-1)[0, :10]
                # print("predictions")
                # print(predictions)
                ev.add_instance(goal, predictions)
        else:
            for sequence, user in self.dataset.validation_set(epochs=1):
                seq_lengths = list(range(1, len(sequence)))  # 1, 2, 3, ... len(sequence)-1
                for seq_length in seq_lengths:
                    X = np.zeros((1, self.max_length, self._input_size()),
                                 dtype=self._input_type)  # input shape of the RNN
                    # Y = np.zeros((1, self.n_items))  # Y가 왜 있는지????? 안쓰임
                    length = np.zeros((1,), dtype=np.int32)

                    seq_by_max_length = sequence[max(length - self.max_length, 0):seq_length]  # last max length or all
                    X[0, :len(seq_by_max_length), :] = np.array(map(lambda x: self._get_features(x), seq_by_max_length))
                    length[0] = len(seq_by_max_length)

                    output = self.sess.run(self.softmax, feed_dict={self.X: X, self.length: length})
                    predictions = np.argpartition(-output, list(range(10)), axis=-1)[0, :10]
                    # print("predictions")
                    # print(predictions)
                    goal = sequence[seq_length:][0]
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
        features have the following structure: [one_hot_encoding, personal_rating on a scale of ten, average_rating on a scale of ten, popularity on a log scale of ten]
        '''

        # item = (item_id, rating)

        one_hot_encoding = np.zeros(self.n_items)
        one_hot_encoding[item[0]] = 1
        return one_hot_encoding