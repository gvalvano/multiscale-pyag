"""
Callback for early stopping.
"""
#  Copyright 2019 Gabriele Valvano
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from idas.callbacks.callbacks import Callback


class EarlyStoppingException(Exception):
    """ Raised to early stop the training. """
    pass


class NeedForTestException(Exception):
    """ Raised to signal the new minimum loss and ask for a test """
    pass


class EarlyStoppingCallback(Callback):
    def __init__(self, min_delta=0.01, patience=20):
        """
         We want to define a minimum acceptable change (min_delta) in the loss function and a patience parameter
        which once exceeded triggers early stopping. When the loss increases or when it stops decreasing by more than
        min_delta, the patience counter activates. Once the patience counter expires, the callback returns a signal
        (stop = True).
        :param min_delta: (float) minimum change required
        :param patience: (int) number of iterations to wait before stopping training
        """
        super().__init__()
        # Define variables here because the callback __init__() is called before the initialization of all variables
        # in the graph.
        assert min_delta > 0
        self.min_delta = min_delta
        self.patience = patience
        self.patience_counter = 0
        self.hist_loss = 1e16
        self.test_on_minimum = False
        self.decay = 0.99997

    def reset(self):
        print('>> Resetting Early stopping callback.')
        self.patience_counter = 0
        self.hist_loss = 1e16
        self.decay = 0.99997

    def on_train_begin(self, training_state, **kwargs):
        """
        test_on_minimum: (bool) if True, returns a signal (NeedForTestException) to ask for a test
        """
        try:
            self.test_on_minimum = kwargs['test_on_minimum']
        except NameError:
            pass

    def on_epoch_end(self, training_state, **kwargs):

        if self.hist_loss - kwargs['es_loss'] >= self.min_delta:
            # update counter and minimum loss
            self.patience_counter = 0
            self.hist_loss = kwargs['es_loss']

            if self.test_on_minimum:
                # signal the need for a test
                raise NeedForTestException
        else:
            self.patience_counter += 1

            if self.patience_counter > self.patience:
                raise EarlyStoppingException

            # decrease best value according to decay
            self.hist_loss /= self.decay
