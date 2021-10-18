import torch
import torch.nn as nn

# Local installs
try:
    import pre_process as proc
    import lstm_cell, prev_states
    from tree import Forest, Tree, TreeNode

# coLab installs
except(ModuleNotFoundError):
    from AI_engineering_management.treelstm import pre_process as proc
    from AI_engineering_management.treelstm import lstm_cell, prev_states
    from AI_engineering_management.treelstm.tree import Forest, Tree, TreeNode

class ChildSumTreeLSTMEncoder(nn.Module):
    """Child-Sum Tree-LSTM Encoder Module."""

    def __init__(self, embed_size, hidden_size,
                 p_keep_input, p_keep_rnn):
        """Create a new ChildSumTreeLSTMEncoder.
        Args:
          embed_size: Integer, number of units in word embeddings vectors.
          hidden_size: Integer, number of units in hidden state vectors.
          p_keep_input: Float, the probability of keeping an input unit.
          p_keep_rnn: Float, the probability of keeping an rnn unit.
        """
        super(ChildSumTreeLSTMEncoder, self).__init__()

        self._embed_size = embed_size
        self._hidden_size = hidden_size

        # Define dropout layer for embedding lookup
        self._drop_input = nn.Dropout(p=1.0 - p_keep_input)

        # Initialize the batch Child-Sum Tree-LSTM cell
        #self.cell = lstm_cell.BatchChildSumTreeLSTMCell(
        #    input_size=embed_size,
        #    hidden_size=hidden_size,
        #    p_dropout=1.0 - p_keep_rnn).cuda()
        self.cell = lstm_cell.BatchChildSumTreeLSTMCell(
            input_size=embed_size,
            hidden_size=hidden_size,
            p_dropout=1.0 - p_keep_rnn)

        # Initialize previous states (to get wirings from nodes on lower level)
        self._prev_states = prev_states.PreviousStates(hidden_size)

    def forward_one(self, tree):
        """Get encoded vector for top node in tree, representing the entire geometry.
        Args:
          Tree structure representing a single STEP file from the dataset
        Returns:
          Output hidden state vector for the final, highest level node in the tree
        """
        tree.resetNodes()

        # Some nodes will be reused at multiple layers of the tree
        # Get these first
        reused_nodes, reused_child_ids, hold_nodes, hold_ids = tree.getReadyNodes([])
        inputs = [(proc.getWordVector(self._embed_size, n.cat_id)) for n in reused_nodes]
        prev_states = self._prev_states.zero_level(len(reused_nodes))
        reused_outputs = self.cell(inputs, prev_states)

        outputs = reused_outputs
        ready_nodes = reused_nodes.copy()

        tree.setDoneNodes(ready_nodes)
        tree.setReadyNodes()
        while(True):
            # Find which nodes should be processed next, which should be held
            ready_nodes, child_ids, hold_nodes, hold_ids = tree.getReadyNodes(ready_nodes)
            
            #hold_0 = torch.zeros(0,self.cell.hidden_size).cuda()
            #hold_1 = torch.zeros(0,self.cell.hidden_size).cuda()
            hold_0 = torch.zeros(0,self.cell.hidden_size)
            hold_1 = torch.zeros(0,self.cell.hidden_size)

            # Concat all held outputs into one vector
            for i in hold_ids:
                hold_0 = torch.cat((hold_0, outputs[0][i].view(1,self.cell.hidden_size)), 0)
                hold_1 = torch.cat((hold_1, outputs[1][i].view(1,self.cell.hidden_size)), 0)

            # Get input word vectors
            inputs = [(proc.getWordVector(self._embed_size, n.cat_id)) for n in ready_nodes]

            # Get previous hidden states
            prev_states = self._prev_states(
                level_nodes=ready_nodes,
                children=child_ids,
                prev_outputs=outputs)

            # Get LSTM outputs for the current layer in the tree
            outputs = self.cell(inputs, prev_states)

            tree.setDoneNodes(ready_nodes)
            if tree.getTreeDone():
                break
            else:
                tree.setReadyNodes()
                ready_nodes.extend(hold_nodes)
                outputs = (torch.cat((outputs[0], hold_0, reused_outputs[0]), 0),
                            torch.cat((outputs[1], hold_1, reused_outputs[1]), 0))

        return outputs[0], outputs[1]

    # Complete forward pass for a full 'forest' containing all trees in a batch
    def forward(self, forest):
        #outputs = torch.zeros(0, self._hidden_size).cuda()
        outputs_0 = torch.zeros(0, self._hidden_size)
        outputs_1 = torch.zeros(0, self._hidden_size)

        for tree in forest.trees:
            out_0, out_1 = self.forward_one(tree)
            outputs_0 = (torch.cat((outputs_0, out_0), 0))
            outputs_1 = (torch.cat((outputs_1, out_1), 0))
        return [outputs_0, outputs_1]

