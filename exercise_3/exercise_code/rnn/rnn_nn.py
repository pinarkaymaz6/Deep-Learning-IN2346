import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, nonlinearity=activation)   
        # Fully connected layer
        #self.fc = nn.Linear(hidden_size, hidden_size)
        #############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size) #out
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size) #hidden
        """
        h_seq = []
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        batch_size = x.size(1)
        # Initializing hidden state for first input using method defined below
        if h is None:
            h = torch.zeros(1, batch_size, self.hidden_size).requires_grad_()
        out, h = self.rnn(x, h.detach())

        #out = out.contiguous().view(-1, self.hidden_size)
        #out = self.fc(out)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return out, h
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        return hidden


class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
    ############################################################################
    # TODO: Build a one layer LSTM with an activation with the attributes      #
    # defined above and a forward function below. Use the nn.Linear() function #
    # as your linear layers.                                                   #
    # Initialse h and c as 0 if these values are not given.                    #
    ############################################################################
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
       
    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h = torch.zeros(1, x.size(0), self.hidden_size).to(device) 
        c = torch.zeros(1, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h, c)) 

        out = self.fc(out[:, -1, :])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
        return h_seq , (h, c)
    

class RNN_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()
    ############################################################################
    #  TODO: Build a RNN classifier                                            #
    ############################################################################
        pass
       
    def forward(self, x):
        pass

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

class LSTM_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128):
        super(LSTM_Classifier, self).__init__()
    ############################################################################
    #  TODO: Build a LSTM classifier                                           #
    ############################################################################
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, 10)
    
    def forward(self, x):
        batch_size = x.shape[1]
        h = torch.zeros(1, batch_size, self.hidden_size) 
        c = torch.zeros(1, batch_size, self.hidden_size)

        out, _ = self.lstm(x, (h, c))
        
        #out_new = out[-1, :, :]
        #out = self.fc(out_new)
        out = self.fc(out)

        return out[-1, :, :]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
        
