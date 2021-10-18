import os
import torch
import random
import time
import math
import numpy as np
from tabulate import tabulate

# Local imports
try:
    import glovar
    from tree import Forest
    import models

# coLab imports
except(ModuleNotFoundError):
    from AI_engineering_management.treelstm.tree import Forest
    from AI_engineering_management.treelstm import glovar
    from AI_engineering_management.treelstm import models

# Utility Functions


def pretty_time(secs):
    """Get a readable string for a quantity of seconds.
    Args:
      secs: Integer, seconds.
    Returns:
      String, nicely formatted.
    """
    if secs < 60.0:
        return '%4.2f secs' % secs
    elif secs < 3600.0:
        return '%4.2f mins' % (secs / 60)
    elif secs < 86400.0:
        return '%4.2f hrs' % (secs / 60 / 60)
    else:
        return '%3.2f days' % (secs / 60 / 60 / 24)


def _dividing_lines():
    # For visuals, when reporting results to terminal.
    return ('--------\t  ----------------\t------------------''\t------------------------')


def _print_epoch_start(epoch, log_path):

    print(_dividing_lines())
    print('Epoch\t\t       loss       \t     accuracy     '
          '\t    train parameters')
    print('        \t  last      avg.  \t  last      avg.  '
          '\t  en rnn       dec rnn   \t')
    print(_dividing_lines())

    log_file = open(log_path, 'a')
    log_file.write(_dividing_lines())
    log_file.write('Epoch\t\t       loss       \t     accuracy     '
          '\t    train parameters')
    log_file.write('        \t  last      avg.  \t  last      avg.  '
          '\t  en rnn       dec rnn   \t')
    log_file.write(_dividing_lines())
    log_file.close()


# Sort dataset into batches of given size
def batchDataset(batch_size, dataset):
    batch_dataset = [Forest()]
    random.shuffle(dataset)

    for tree in dataset:
        if len(batch_dataset[-1].trees) == batch_size:
            batch_dataset.append(Forest())
        batch_dataset[-1].addTree(tree)

    return batch_dataset

class Trainer:

    def __init__(self, model, history, ckpt_dir, train_data=None, test_data=None, val_data=None, best=True):
        """Create a new training wrapper.
        Args:
          model: any model to be trained, be it TensorFlow or PyTorch.
          history: histories.History object for storing training statistics.
          train_data: the data to be used for training.
          test_data: the data to be used for tuning.
          ckpt_dir: String, path to checkpoint file directory.
        """
        self.model = model
        self.history = history
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.num_epochs = 5
        self.stop = False
        if train_data:
            self.batches_per_epoch = len(train_data)
        self.ckpt_dir = ckpt_dir
        self.starting = True
        # Load the latest checkpoint if necessary
        if self.history.global_step > 1:
            if val_data:
                if best:
                    print('Loading best weights...')
                    self._load_best()
                else:
                    print('Loading last weights...')
                    self._load_last()
            else:
                print('Loading last checkpoint...')
                self._load_last()

        # if initialising multi-feature model, load pre-trained encoder
        elif type(self.model) == models.FeatureRecognitionModel:
            print('Loading weights for pre-trained encoder...')
            model_dict = model.state_dict()
            prtr_path = os.path.join(self.ckpt_dir, '%s_%s' % (self.model.pretrained_model, 'best'))
            prtr_dict = torch.load(prtr_path)
            prtr_dict = {k: v for k, v in prtr_dict.items() if 'encoder' in k}
            model_dict.update(prtr_dict)
            model.load_state_dict(model_dict)

        if not train_data and not test_data:
            if not val_data:
                print('No dataset.')
        elif not val_data:
            if not train_data or not test_data:
                print('Train and test datasets required for training.')
        #self.model.cuda()
        self.log_path = os.path.join(glovar.PKL_DIR, history.name + '.txt')

    # Save latest weights as checkpoint
    def _checkpoint(self, is_best):

        # Save weights under epoch number
        file_path = self.ckpt_path(self.history.global_epoch - 1)
        torch.save(self.model.state_dict(), file_path)

        # Save weights as latest
        file_path = self.ckpt_path(-2)
        torch.save(self.model.state_dict(), file_path)

        # If these are the best weights, save them as best
        if is_best:
            #print('Saving checkpoint with new best tuning accuracy...')
            file_path = self.ckpt_path(-1)
            torch.save(self.model.state_dict(), file_path)

    # Set path to save checkpoint (weights)
    def ckpt_path(self, marker):
        if marker == -1:
            marker = 'best'
        elif marker == -2:
            marker = 'latest'
        else:
            marker = str(marker)

        return os.path.join(
            self.ckpt_dir,
            '%s_%s' % (self.model.name, marker))

    # End training epoch
    def _end_epoch(self):
        self._epoch_end = time.time()
        time_taken = self._epoch_end - self._epoch_start
        avg_time, avg_loss, change_loss, avg_acc, change_acc, is_best = \
            self.history.end_epoch(time_taken)
        #self._report_epoch(avg_time)
        self._checkpoint(is_best)
        self.history.save()

    # End training step
    def _end_step(self, loss, acc):
        self.step_end = time.time()
        time_taken = self.step_end - self.step_start
        global_step, avg_time, avg_loss, avg_acc = \
            self.history.end_step(time_taken, loss, acc)
        self._report_step(global_step, loss, avg_loss, acc, avg_acc, avg_time)

    # Load best weights
    def _load_best(self):
        file_path = self.ckpt_path(-1)
        self.model.load_state_dict(torch.load(file_path))

    # Load latest weights
    def _load_last(self):
        file_path = self.ckpt_path(-2)
        self.model.load_state_dict(torch.load(file_path))

    @property
    def progress_percent(self):
        percent = (self.history.global_step % self.batches_per_epoch) \
                  / self.batches_per_epoch \
                  * 100
        rounded = int(np.ceil(percent / 10.0) * 10)
        return rounded

    def _report_epoch(self, avg_time):
        _print_dividing_lines()
        print('\t\t\t\t\t\t\t%s'
              % pretty_time(np.average(avg_time)))

    @property
    def report_every(self):
        return int(np.floor(self.batches_per_epoch / 10))

    # Report model performance and parameters
    def _report_step(self, global_step, loss, avg_loss, acc, avg_acc, avg_time):
        if global_step % self.batches_per_epoch == 0:
            out_str = ('%s\t\t'
                  '%8.4f  %8.4f\t'
                  '%6.4f%%  %6.4f%%\t'
                  '    %s\t\t'
                  ' %s'
                  % (int(global_step / self.batches_per_epoch),
                     loss,
                     avg_loss,
                     acc[-1] * 100,
                     avg_acc * 100,
                     self.model.train_state[0],
                     self.model.train_state[1]))
            print(out_str)
            log_file = open(self.log_path, 'a')
            log_file.write('\n' + out_str)
            log_file.close()

    # Start training epoch
    def _start_epoch(self):
        if (self.history.global_step - 1) % (self.batches_per_epoch * 10) == 0 or self.starting:
            _print_epoch_start(self.history.global_epoch, self.log_path)
            self.starting = False
        self.model.train()
        self._epoch_start = time.time()

    # Start training step
    def _start_step(self):
        self.step_start = time.time()

    # Single training step (forward pass + optimisation for a single batch)
    def step(self, batch):
        self.model.zero_grad()
        _, loss, acc = self.model.forward(batch)
        self.model.optimize(loss)
        #return loss.cpu().data.numpy(), acc
        return loss.data.numpy(), acc

    @property
    def steps_remaining(self):
        return self.batches_per_epoch \
               - (self.history.global_step % self.batches_per_epoch)

    def _stopping_condition_met(self, start_epoch):
        if (self.history.global_epoch - start_epoch) == self.num_epochs:
            self.stop = True
        return self.stop

    # Run the training algorithm
    def train(self, num_epochs=5, train_strat=[5,1,0]):
        self.num_epochs = num_epochs
        start_epoch = self.history.global_epoch

        if type(self.model) == models.ClassificationModel:
            while not self._stopping_condition_met(start_epoch):
                self.training_step()

        else:
            for i in range(3):
                counter = 0
                self.model.set_train_params(params=i)
                while counter < train_strat[i]:
                    self.training_step()
                    counter += 1

        print('Best weights: epoch {}'.format(self.history.best))
        log_file = open(self.log_path, 'a')
        log_file.write('\n\nBest weights: epoch {}\n\n'.format(self.history.best))
        log_file.close()

    # Perform a full training epoch (training step for each batch then tuning step)
    def training_step(self):
        self._start_epoch()
        for batch in self.train_data:
            self._start_step()
            loss, acc = self.step(batch)
            self._end_step(loss, acc)
        self._tuning()
        self._end_epoch()

    # Run the validation algorithm
    def validate(self):
        print('Running model on validation dataset')
        cum_loss = 0
        i = 0
        if type(self.model) == models.FeatureRecognitionModel:
            cum_acc = [0,0,0]
            limit = 25
            err_count = np.zeros(self.model.max_output-1)
            close_count = np.zeros(self.model.max_output-1)
            corr_count = np.zeros(self.model.max_output-1)
        else:
            cum_acc = 0
            limit = 24
        err_mat = np.zeros((limit,limit))

        errors = []
        with torch.no_grad():
            for batch in self.val_data:
                pred, loss, acc, err_mat, errs = self.model.forward_val(batch, err_mat)
                try:
                    for j in range(3):
                        cum_acc[j] += acc[j]
                except IndexError:
                    cum_acc += acc
                cum_loss += loss
                for e in errs:
                    errors.append(e)
                i += 1
                if i % 100 == 0:
                    print('{} Batches complete | Loss: {} | Accuracy: {}'.format(
                        i, cum_loss/i, cum_acc/i))
        
        # Print table of predictions against ground truth
        col_headers = ['l'] + ['p={}'.format(j) for j in range(limit)]
        row_headers = np.array([[j for j in range(limit)]])
        m = np.concatenate((row_headers.T, err_mat), axis=1)
        table = tabulate(m, col_headers, tablefmt='fancy_grid')
        print(table)
        print('\n\n')

        # Print table of number of instances, % correct for each feature class
        sum_mat = np.zeros((limit, 3))
        for i in range(limit):
            tot = int(sum(err_mat[i]))
            perc_corr = err_mat[i][i] / tot * 100
            sum_mat[i] = [i, tot, perc_corr]
        col_headers = ['feature class', 'num in dataset', '% correct']
        table = tabulate(sum_mat, col_headers, tablefmt='fancy_grid')
        print(table)

        # Print validation loss and accuracy scores
        loss = cum_loss / len(self.val_data)
        print('Validation loss: %5.3f' % loss)
        try:
            for j in range(3):
                acc[j] = cum_acc[j] / len(self.val_data)
            print('Validation precision: %5.2f%%' % (acc[0] * 100))
            print('Validation recall: %5.2f%%' % (acc[1] * 100))
            print('Validation f-score: %5.2f%%' % (acc[2] * 100))
        except IndexError:
            acc = cum_acc / len(self.val_data)
            print('Validation accuracy: %5.2f%%' % (acc * 100))

        # Print list of models with incorrect classifications
        err_print = '\nList of incorrectly classified models (not close):'
        close_print = '\nList of incorrectly classified models (close):'
        for e in errors:
            if type(self.model) == models.FeatureRecognitionModel:
                print_str = e[1] + ": predicted as " + e[2]
                if e[0] == 0:
                    err_count[e[3]-1] += 1
                    err_print += '\n' + print_str
                elif e[0] == 1:
                    close_count[e[3]-1] += 1
                    close_print += '\n' + print_str
                else:
                    corr_count[e[3]-1] += 1
            else:
                print(e)
        print(err_print)
        print(close_print)

        # Print % of models correctly classified: overall and by number of features
        if type(self.model) == models.FeatureRecognitionModel:
            tot_corr = np.sum(corr_count)
            tot_err = np.sum(err_count)
            tot_close = np.sum(close_count)
            print('\nCorrect predictions: {}'.format(tot_corr))
            print('Incorrect predictions: {}'.format(tot_err + tot_close))
            print('\nTotal: {:.2f}% correct'.format(100*(tot_corr/(tot_corr+tot_err + tot_close))))
            print('{:.2f}% correct or close'.format(100*((tot_corr+tot_close)/(tot_corr+tot_err+tot_close))))
            for i in range(len(err_count)):
                print_val_1 = 100*corr_count[i]/(corr_count[i] + err_count[i] + close_count[i])
                print_val_2 = 100*(corr_count[i] + close_count[i])/(corr_count[i] + err_count[i] + close_count[i])
                print('{} features: {:.2f}% correct, {:.2f}% close or correct'.format(i+1, print_val_1, print_val_2))

    # Perform tuning step
    def _tune(self, test_data):

        cum_loss = 0
        try:
            cum_acc = [0,0,0]
            tuning_acc = [0,0,0]
            for batch in test_data:
                _, loss, acc = self.model.forward(batch)
                for i in range(3):
                    cum_acc[i] += acc[i]
                cum_loss += loss
            for i in range(3):
                tuning_acc[i] = cum_acc[i] / len(test_data)
        except IndexError:
            cum_acc = 0
            for batch in test_data:
                _, loss, acc = self.model.forward(batch)
                cum_acc += acc
                cum_loss += loss
            tuning_acc = cum_acc / len(test_data)
        tuning_loss = cum_loss / len(test_data)
        avg_acc, change_acc = self.history.end_tuning(tuning_acc, tuning_loss)
        if self.stop:
            print('Tuning accuracy: %5.2f%%' % (tuning_acc * 100))
            print('Average tuning accuracy: %5.2f%% (%s%5.3f%%)' %
              (avg_acc * 100,
               '+' if change_acc > 0 else '',
               change_acc * 100))
            log_file = open(self.log_path, 'a')
            log_file.write('\nTuning accuracy: %5.2f%%' % (tuning_acc * 100))
            log_file.write('\nAverage tuning accuracy: %5.2f%% (%s%5.2f%%)' %
              (avg_acc * 100,
               '+' if change_acc > 0 else '',
               change_acc * 100))
            log_file.close()
            self.stop = False

    def _tuning(self):
        with torch.no_grad():
            self._tune(self.test_data)

