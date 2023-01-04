import numpy as np
import logging
import math
import torch
import torch.nn as nn
import pdb
LOGGER = logging.getLogger(__name__)

def MSE (Y, Yhat, mask):
    return torch.sum(torch.mul(torch.square(Y-Yhat), mask)) / (torch.sum(mask) + 1)

class EarlyStopping():
    """
    Early stopping is used to stop the training when the loss does not decrease after a certain number of epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        patience: how many epochs to wait before stopping when the loss is not improving
        min_delta: the minimum difference between new loss and old loss for new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
                
class GRU(nn.Module):
    '''
    This is the simplest GRU model, where GRU directly learns the mapping between input and output.
    paramters: 
        ninp: The number of expected features in the input.
        nhid : The number of features in the hidden state h
        nlayers : Number of recurrent layers.
        dropout:  If it is non-zero, introduce a dropout layer on the outputs of each GRU layer except the last layer. Default 0.
        nout: the number of expected variables in the outputs.
        batch_first: If True, the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). 


        forward(self, W_inputs, stat_ini_sq, hidden) shows how we build the model architecture. N2OGRU has one
        GRU blocks which learns mapping between inputs and outputs.
        accpets the inputs to predict ouput. 
    '''    
    def __init__(self, ninp, nhid, nlayers, nout, dropout, batch_first=True):
        super(GRU, self).__init__()
        self.gru = nn.GRU(ninp,nhid,nlayers,dropout=dropout,batch_first=batch_first)
        self.densor = nn.Linear(nhid, nout)
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop=nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        # the weights of the model is initialized as uniform(-0.1, 0.1). 
        initrange = 0.1 #may change to a small value
        self.densor.bias.data.zero_()
        self.densor.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, hidden_head):
        hiddens, hidden_tail = self.gru(inputs, hidden_head)
        output = self.densor(self.drop(hiddens))
        return output, hiddens
    
    
    def init_hidden(self, bsz):
        # Generate the zero hidden states.
        # bsz: batch size
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)
    
    
class Transformer(nn.Module):
    """Transformer model class, which relies on PyTorch's TransformerEncoder class. This class implements the encoder of a transformer network which can be used for regression.
    Unless the number of inputs is divisible by the number of transformer heads (``transformer_nheads``), it is
    necessary to use an embedding network that guarantees this. 
    This class is adopted from the Nueralhydrology project. 
    Their github : https://github.com/neuralhydrology/neuralhydrology/blob/master/neuralhydrology/modelzoo/transformer.py
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'encoder', 'head']

    def __init__(self, d_model: int, n_embed:int, nhead: int, dim_feedforward: int,
                 nlayers: int, nout: int, dropout: float = 0.5, batch_first=False, positional_encoding_type='sum'):
        
        super(Transformer, self).__init__()

        # embedding net before transformer
        self.embedding_net = nn.Linear(d_model, n_embed)

        # ensure that the number of inputs into the self-attention layer is divisible by the number of heads
        if n_embed % nhead != 0:
            raise ValueError("Embedding dimension must be divisible by number of transformer heads. "
                             "Use statics_embedding/dynamics_embedding and embedding_hiddens to specify the embedding.")

        self._sqrt_embedding_dim = math.sqrt(n_embed)

        # positional encoder
        self._positional_encoding_type = positional_encoding_type
        if self._positional_encoding_type.lower() == 'concatenate':
            encoder_dim = n_embed * 2
        elif self._positional_encoding_type.lower() == 'sum':
            encoder_dim = n_embed
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {self._positional_encoding_type}")
        self.positional_encoder = _PositionalEncoding(embedding_dim=n_embed,
                                                      dropout=dropout,
                                                      position_type=self._positional_encoding_type)

        # positional mask
        self._mask = None

        # encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=encoder_dim,
                                                    nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers,
                                             num_layers=nlayers,
                                             norm=None)

        # head (instead of a decoder)
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(encoder_dim, nout)
        # init weights and biases
        self._reset_parameters()

    def _reset_parameters(self):
        # this initialization strategy was tested empirically but may not be the universally best strategy in all cases.
        initrange = 0.1
        for layer in self.encoder.layers:
            layer.linear1.weight.data.uniform_(-initrange, initrange)
            layer.linear1.bias.data.zero_()
            layer.linear2.weight.data.uniform_(-initrange, initrange)
            layer.linear2.bias.data.zero_()
            
        self.head.bias.data.zero_()
        self.head.weight.data.uniform_(-initrange, initrange)

    def forward(self, data: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """Perform a forward pass on a transformer model without decoder.

        Parameters
        ----------
        data : torch.Tensor, containing input features.

        Returns
        -------
        torch.Tensor
            Model outputs, `y_hat`: model predictions of shape [sequence length,batch size, number of target variables].
        """
        # pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data)

        positional_encoding = self.positional_encoder(x_d * self._sqrt_embedding_dim)
        
        # mask out future values
        if self._mask is None or self._mask.size(0) != len(x_d):
            self._mask = torch.triu(x_d.new_full((len(x_d), len(x_d)), fill_value=float('-inf')), diagonal=1)
        
        # encoding
        output = self.encoder(positional_encoding, self._mask)

        # head, from [sequence length, batch size, number of target variables] to [batch size, sequence length, number of target variables]
        pred = self.head(self.dropout(output.transpose(0, 1)))
        pred = pred.transpose(0, 1)

        return pred, x_d, positional_encoding


class _PositionalEncoding(nn.Module):
    """Class to create a positional encoding vector for timeseries inputs to a model without an explicit time dimension.

    This class implements a sin/cos type embedding vector with a specified maximum length. Adapted from the PyTorch
    example here: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Parameters
    ----------
    embedding_dim : int
        Dimension of the model input, which is typically output of an embedding layer.
    dropout : float
        Dropout rate [0, 1) applied to the embedding vector.
    max_len : int, optional
        Maximum length of positional encoding. This must be larger than the largest sequence length in the sample.
    """

    def __init__(self, embedding_dim, position_type, dropout, max_len=5000):
        super(_PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, int(np.ceil(embedding_dim / 2) * 2))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(max_len * 2) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[:, :embedding_dim].unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        if position_type.lower() == 'concatenate':
            self._concatenate = True
        elif position_type.lower() == 'sum':
            self._concatenate = False
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {position_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for positional encoding. Either concatenates or adds positional encoding to encoder input data.

        Parameters
        ----------
        x : torch.Tensor
            Dimension is ``[sequence length, batch size, embedding output dimension]``.
            Data that is to be the input to a transformer encoder after including positional encoding.
            Typically this will be output from an embedding layer.

        Returns
        -------
        torch.Tensor
            Dimension is ``[sequence length, batch size, encoder input dimension]``.
            The encoder input dimension is either equal to the embedding output dimension (if ``position_type == sum``)
            or twice the embedding output dimension (if ``position_type == concatenate``).
            Encoder input augmented with positional encoding.

        """
        if self._concatenate:
            x = torch.cat((x, self.pe[:x.size(0), :].repeat(1, x.size(1), 1)), 2)
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)