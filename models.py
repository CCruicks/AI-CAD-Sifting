import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import itertools

# Local install
try:
    import encoder, decoder

# coLab install
except(ModuleNotFoundError):
    from AI_engineering_management.treelstm import encoder, decoder


FRAMEWORKS = ['tf', 'torch']
_DEFAULT_CONFIG = {
    'batch_size': 32,
    'hidden_size': 300,
    'projection_size': 200,
    'learning_rate': 1e-3,
    'grad_clip_norm': 0.0,
    '_lambda': 0.0,
    'p_keep_input': 0.9,
    'p_keep_rnn': 0.9,
    'p_keep_fc': 0.9,
    'tune_embeddings': True
}


class Config:
    """Wrapper of config variables."""

    def __init__(self, default=_DEFAULT_CONFIG, **kwargs):
        """Create a new Config.
        Args:
          default: Dictionary of default values. These can be passed in, or else
            the _DEFAULT_CONFIG from this file will be used.
        """
        self.default = default
        self.kwargs = kwargs
        self.batch_size = self._value('batch_size', kwargs)
        self.hidden_size = self._value('hidden_size', kwargs)
        self.projection_size = self._value('projection_size', kwargs)
        self.learning_rate = self._value('learning_rate', kwargs)
        self.grad_clip_norm = self._value('grad_clip_norm', kwargs)
        self._lambda = self._value('_lambda', kwargs)
        self.p_keep_input = self._value('p_keep_input', kwargs)
        self.p_keep_rnn = self._value('p_keep_rnn', kwargs)
        self.p_keep_fc = self._value('p_keep_fc', kwargs)
        self.tune_embeddings = self._value('tune_embeddings', kwargs)
        for key in [k for k in kwargs.keys()
                    if k not in self.default.keys()]:
            setattr(self, key, kwargs[key])

    def __delitem__(self, key):
        pass

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __repr__(self):
        x = 'Config as follows:\n'
        for key in sorted(self.keys()):
            x += '\t%s \t%s%s\n' % \
                 (key, '\t' if len(key) < 15 else '', self[key])
        return x

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def dropout_keys(self):
        return [k for k in self.__dict__.keys() if k.startswith('p_keep_')]

    def keys(self):
        return [key for key in self.__dict__.keys()
                if key not in ['default', 'kwargs']]

    def to_json(self):
        return dict(self.__dict__)

    def _value(self, key, kwargs):
        if key in kwargs.keys():
            return kwargs[key]
        else:
            return self.default[key]


class Model:
    """Base class for a model of any kind."""

    def __init__(self, framework, config):
        """Create a new Model.
        Args:
          framework: String, the framework of the model, e.g. 'pytorch'.
          config: Config object, a configuration settings wrapper.
        """
        self.framework = framework
        self.config = config
        for key in config.keys():
            setattr(self, key, config[key])

    def accuracy(self, *args):
        raise NotImplementedError

    def forward(self, *args):
        """Forward step of the network.
        Returns:
          predictions, loss, accuracy.
        """
        raise NotImplementedError

    def logits(self, *args):
        raise NotImplementedError

    def loss(self, *args):
        raise NotImplementedError

    def optimize(self, *args):
        raise NotImplementedError

    def predictions(self, *args):
        raise NotImplementedError


class PyTorchModel(Model, nn.Module):
    """Base for a PyTorch model."""

    def __init__(self, name, config, embedding_matrix):
        Model.__init__(self, 'pytorch', config)
        nn.Module.__init__(self)

        self.name = name

        # Define embedding.
        self.embedding = embedding_matrix
        self.embed_size = embedding_matrix.size()[0]
        #self.embedding.cuda()

        # Define loss
        #self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.criterion = torch.nn.CrossEntropyLoss()

    @staticmethod
    def accuracy(correct_predictions, batch_size):
        # batch_size may vary - i.e. the last batch of the data set.
        #correct = correct_predictions.cpu().sum().data.numpy()
        correct = correct_predictions.sum().data.numpy()
        return correct / float(batch_size)

    def _biases(self):
        return [p for n, p in self.named_parameters() if n in ['bias']]

    @staticmethod
    def correct_predictions(predictions, labels):
        return predictions.eq(labels)

    def forward(self, forest):
        # Need to return predictions, loss, accuracy
        raise NotImplementedError

    def logits(self, forest):
        raise NotImplementedError

    def loss(self, logits, labels):
        loss = self.criterion(logits, labels)
        return loss

    def optimize(self, loss):
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def predictions(logits):
        return logits.max(1)[1]

    def _weights(self):
        return [p for n, p in self.named_parameters() if n in ['weight']]

    def zero_grad(self):
        self.optimizer.zero_grad()


class ClassificationModel(PyTorchModel):
    """Single feature classification model."""

    def __init__(self, name, config, embedding_matrix):
        super(ClassificationModel, self).__init__(name, config, embedding_matrix)

        # Define encoder.
        self.encoder = encoder.ChildSumTreeLSTMEncoder(
            self.embed_size, self.hidden_size,
            self.p_keep_input, self.p_keep_rnn)

        # Define MLP
        #self.logits_layer = nn.Linear(self.hidden_size, 24).cuda()
        self.logits_layer = nn.Linear(self.hidden_size, 24)

        # Define optimizer
        params = [{'params': self.encoder.cell.parameters()},
                  {'params': self.logits_layer.parameters()}]
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        # Initialize parameters
        nn.init.xavier_uniform_(self.logits_layer.weight.data, gain=1.)

    @staticmethod
    def current_batch_size(forest):
        return len(forest.trees)

    # Complete a forward pass through the network during training
    # Return list of predicted classes, loss and accuracy for the forest
    def forward(self, forest):
        #labels = Variable(
        #    torch.from_numpy(np.array(forest.classes)).long(),
        #    requires_grad=False).cuda()
        classes = [c[0] for c in forest.classes]
        labels = Variable(
            torch.from_numpy(np.array(classes)).long(),
            requires_grad=False)
        logits = self.logits(forest)
        loss = self.loss(logits, labels)
        predictions = self.predictions(logits).type_as(labels)
        correct = self.correct_predictions(predictions, labels)
        accuracy = self.accuracy(correct, self.current_batch_size(forest))
        return predictions, loss, accuracy

    # Complete a forward pass through the network during validation
    # Return predicted classes, loss, accuracy, matrix of predictions vs truth and list of errors
    def forward_val(self, forest, error_mat):
        #labels = Variable(
        #    torch.from_numpy(np.array(forest.classes)).long(),
        #    requires_grad=False).cuda()
        classes = [c[0] for c in forest.classes]
        filenames = [t.name for t in forest.trees]
        errors = []
        labels = Variable(
            torch.from_numpy(np.array(classes)).long(),
            requires_grad=False)
        logits = self.logits(forest)
        loss = self.loss(logits, labels)
        predictions = self.predictions(logits).type_as(labels)
        correct = self.correct_predictions(predictions, labels)
        for p, l, n in zip(predictions, labels, filenames):
            error_mat[l][p] += 1
            if not p == l:
                errors.append(n)
        accuracy = self.accuracy(correct, self.current_batch_size(forest))
        return predictions, loss, accuracy, error_mat, errors

    # Compute output vector for each tree in the forest
    def logits(self, forest):
        encodings = self.encoder.forward(forest)
        logits = self.logits_layer(encodings)
        return logits

class FeatureRecognitionModel(PyTorchModel):
    """Multi-feature recognition model."""

    def __init__(self, name, config, embedding_matrix, pretrained_model, train_en=False, max_features=5):
        super(FeatureRecognitionModel, self).__init__(name, config, embedding_matrix)

        # Set fixed number of labels to predict per tree
        self.max_output = max_features + 1

        # Load pre-trained weights for the encoder
        self.pretrained_model = pretrained_model

        # Define encoder.
        self.encoder = encoder.ChildSumTreeLSTMEncoder(
            self.embed_size, self.hidden_size,
            self.p_keep_input, self.p_keep_rnn)

        # Define decoder.
        self.decoder = decoder.DecoderRNN(25, self.hidden_size)

        # Define optimizer
        self.set_train_params()

    # Calculate average loss per tree in a forest
    def calculate_loss(self, forest, outputs):
        batch_size = self.current_batch_size(forest)
        cum_loss = 0
        np_labels = np.zeros((batch_size, self.max_output))
        init_labels = Variable(torch.from_numpy(np_labels).long(), requires_grad=False)
        predictions = self.predictions(outputs).type_as(init_labels)

        for i in range(batch_size):

            pred = predictions[i].tolist()
            if 24 in pred:
                num_labels = pred.index(24) + 1
            else:
                num_labels = self.max_output

            # Order of outputs irrelevent- try every possible order of ground truth
            class_list = list(itertools.permutations(forest.classes[i]))
            batch_labels = np.full((len(class_list), self.max_output), 24)
            losses = torch.zeros(len(class_list))

            # Calculate loss for each ground truth sequence
            for j in range(len(class_list)):
                c = class_list[j]
                batch_labels[j][:len(c)] = c
                #target = Variable(torch.from_numpy(np.array(batch_labels[j])).long(), requires_grad=False)
                target = Variable(torch.from_numpy(np.array(batch_labels[j][:num_labels])).long(), requires_grad=False)
                #target = target.view(1, self.max_output)
                target = target.view(1, num_labels)
                #losses[j] = self.loss(outputs[i].view(1, 25, self.max_output), target)
                losses[j] = self.loss(outputs[i][:, :num_labels].view(1, 25, num_labels), target)

            # Take the lowest loss - when true sequence in same order as predicted
            min_loss, min_index = torch.min(losses, 0)
            cum_loss += min_loss
            np_labels[i] = batch_labels[min_index]

        labels = Variable(torch.from_numpy(np_labels).long(), requires_grad=False)

        return cum_loss / batch_size, labels

    # Calculate precision, recall and f-score
    def calculate_acc(self, predictions, labels):
        #cum_ap = 0

        # True positives, false positives
        tp = 0
        fp = 0

        num_truth = 0
        for i in range(len(predictions)):
            pred = predictions[i].tolist()
            lab = labels[i].tolist()

            #num_truth = 0
            num_pred = 0
            l_break = False
            p_break = False
            for l, p in zip(lab, pred):
                if l == 24:
                    l_break = True
                    #num_truth += 1
                if p == 24:
                    p_break = True
                    #num_pred += 1
                if l_break and p_break:
                    break
                if not l_break:
                    num_truth += 1
                if not p_break:
                    num_pred += 1

            #precision = []
            #recall = []
            #tp = 0
            #fp = 0

            # Compare predictions to ground truth
            for j in range(num_pred):
                if pred[j] in lab:
                    lab.remove(pred[j])
                    tp += 1
                else:
                    fp += 1
                #precision.append(tp / (tp + fp))
                #recall.append(tp / num_truth)
        try:
            precision = tp / (tp + fp)
        except ZeroDivisionError:
            precision = 0

        recall = tp / num_truth

        # f-score: weighted average of precision and recall
        try:
            fscore = (2 * precision * recall) / (precision + recall)
        except ZeroDivisionError:
            fscore = 0

            #for j in range(len(recall) - 1, 0, -1):
            #    if recall[j] > recall[j - 1]:
            #        precision[j - 1] = precision[j]
            #    else:
            #        del recall[j]
            #        del precision[j]

            #cum_ap += sum(precision) / len(precision)

        #ap = cum_ap / len(predictions)
        return precision, recall, fscore


    @staticmethod
    def current_batch_size(forest):
        return len(forest.trees)

    # Complete a forward pass through the network during training
    # Return list of predicted classes, loss and accuracy scores for the forest
    def forward(self, forest):
        logits = self.logits(forest)
        loss, labels = self.calculate_loss(forest, logits)
        predictions = self.predictions(logits).type_as(labels)
        precision, recall, fscore = self.calculate_acc(predictions, labels)
        return predictions, loss, [precision, recall, fscore]

    # Complete a forward pass through the network during validation
    # Return predicted classes, loss, accuracy, matrix of predictions vs truth and list of errors
    def forward_val(self, forest, error_mat):
        #labels = Variable(
        #    torch.from_numpy(np.array(forest.classes)).long(),
        #    requires_grad=False).cuda()
        classes = forest.classes
        filenames = [t.name for t in forest.trees]
        errors = []
        logits = self.logits(forest)
        loss, labels = self.calculate_loss(forest, logits)
        predictions = self.predictions(logits).type_as(labels)
        precision, recall, fscore = self.calculate_acc(predictions, labels)
        sim_classes = [[1,2], [5,6], [7,8], [9,10], [12,13], [15,16]]
        for i in range(len(filenames)):
            all_correct = True
            all_close = True
            pred_out = ""
            np_pred = np.array(predictions[i])
            np_lab = np.array(labels[i])
            for p, l in zip(np_pred, np_lab):
                error_mat[l][p] += 1
                if not p == l:
                    all_correct = False
                    if not ([l,p] in sim_classes or [p,l] in sim_classes):
                        all_close = False
                if p == 24:
                    break
                pred_out += (str(p) + "_")
            pred_out = pred_out.strip("_")
            if not all_correct:
                if not all_close:
                    errors.append([0, filenames[i], pred_out, np.where(np_lab==24)[0][0]])
                else:
                    errors.append([1, filenames[i], pred_out, np.where(np_lab==24)[0][0]])
            else:
                errors.append([2, filenames[i], pred_out, np.where(np_lab==24)[0][0]])

        return predictions, loss, [precision, recall, fscore], error_mat, errors


    # Compute output vector for each tree in the forest
    def logits(self, forest):
        encodings = self.encoder.forward(forest)
        logits = self.decoder.forward(encodings, self.max_output)
        return logits

    # Set which parameters to tune during the next round of training
    def set_train_params(self, params=0):

        # Train all encoder and decoder paramaters
        if params == 1:
            params = [{'params': self.encoder.cell.parameters()},
                      {'params': self.decoder.lstm_cell.parameters()},
                      {'params': self.decoder.fc_out.parameters()}]
            self.train_state = ['yes', 'yes']

        # Train all decoder parameters
        elif params == 0:
            params = [{'params': self.decoder.lstm_cell.parameters()},
                      {'params': self.decoder.fc_out.parameters()}]
            self.train_state = ['no', 'yes']

        # Train the output fully connected layer parameters only
        else:
            params = [{'params': self.decoder.fc_out.parameters()}]
            self.train_state = ['no', 'no']
        
        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)

