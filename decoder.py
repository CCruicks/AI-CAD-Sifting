import torch
import torch.nn as nn

class DecoderRNN(nn.Module):

    def __init__(self, vocab_size, hidden_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        # define properties
        #self.hidden_size = hidden_size * 2
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # attention
        #self.attn = nn.Linear(hidden_size * 2, 1)

        # lstm cell
        self.lstm_cell = nn.LSTMCell(input_size=vocab_size, 
                hidden_size=self.hidden_size)

        # pooling layer
        #self.pool = nn.MaxPool1d(2, stride=2)

        # output fc layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=vocab_size)
        nn.init.xavier_uniform_(self.fc_out.weight.data, gain=1.)

        # activations
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, max_length):

        # batch size
        batch_size = features[0].size(0)
        
        # define output tensor
        outputs = torch.empty((batch_size, self.vocab_size, max_length))

        # init hidden and cell states
        #hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
        hidden_state = features[1]
        #hs_1 = torch.zeros((batch_size, int(self.hidden_size/2)))

        #cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()
        cell_state = torch.zeros((batch_size, self.hidden_size))
        #cs_1 = torch.zeros((batch_size, int(self.hidden_size/2)))

        # init input vector to zeros
        input_vector = torch.zeros(batch_size, self.vocab_size)

        for t in range(max_length):

            #hidden_state = torch.cat((features[1], hs_1), 1)
            #cell_state = torch.cat((features[0], cs_1), 1)

            #hidden_state, cell_state = self.lstm_cell(input_vector, (hidden_state, cell_state))
            hidden_state, cell_state = self.lstm_cell(input_vector, (features[1], cell_state))

            out = self.fc_out(hidden_state)
            #hs_1 = torch.squeeze(self.pool(hidden_state.unsqueeze(0)))
            #cs_1 = torch.squeeze(self.pool(cell_state.unsqueeze(0)))

            # build the output tensor
            outputs[:, :, t] = out

            #input_vector = out

            # build input vector for the next cell by adding the previous output
            max_indices = torch.argmax(out, dim=1)
            input_update = input_vector.clone()
            for i in range(batch_size):
                for j in range(self.vocab_size):
                    if j == max_indices[i]:
                        input_update[i][j] = input_update[i][j] + 1

            input_vector = input_update

        return outputs

