"""
    Author:         Yassin Riyazi
    Date:           04-08-2025
    Description:    Train an embedding-based LSTM for time series data.

    TODO:
        -

"""
import torch 
from .AutoEncoder_CNN import Autoencoder_CNN
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        """
        Initializes the LSTM model.
        Args:
            input_dim (int): Dimension of the input space
            hidden_dim (int): Dimension of the hidden state in LSTM
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate for LSTM layers
        Returns:
            None: Initializes the LSTM layer and a fully connected layer.

        """
        super(LSTMModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True, device=self.device)
        self.fc = nn.Linear(hidden_dim, 1, device=self.device)  # output layer

        self.h = None
        self.c = None

    def forward(self, x: torch.Tensor,
                ) -> torch.Tensor:
        """
        Forward pass through the LSTM model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1)
        """
        if self.h is None or self.c is None:
            self.reset_states(x)
        
        out, _ = self.lstm(x, (self.h, self.c))
        out = out[:, -1, :]  # take the last hidden state
        out = self.fc(out)
        return out
    
    def reset_states(self,x: torch.Tensor) -> None:
        """
        Resets the hidden and cell states of the LSTM.
        Args:
            x (torch.Tensor): Input tensor to determine batch size  
        Returns:
            None: Resets the states to zero.
        """
        self.h = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=self.device)
        self.c = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=self.device)


class Encoder_LSTM(torch.nn.Module):
    def __init__(self,
                 address_autoencoder:str,
                 input_dim:int   = 1000,  # input dimension
                 hidden_dim:int  = 128 ,  # hidden dimension
                 num_layers:int  = 2   ,  # number of LSTM layers
                 dropout:float   = 0.2 ,  # dropout rate
                 ) -> None:
        """
        Initializes the LSTM encoder.
        Args:
            address_autoencoder (str): Path to the pre-trained autoencoder model
            input_dim (int): Dimension of the input space
            hidden_dim (int): Dimension of the hidden state in LSTM
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate for LSTM layers

        Returns:
            None: Initializes the LSTM layer.
        """
        super(Encoder_LSTM, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Properties = {
                            "input_dim": input_dim,
                            "hidden_dim": hidden_dim,
                            "num_layers": num_layers,
                            "dropout": dropout
                        }

        self.load_autoencoder(address_autoencoder, embedding_dim=input_dim)


        self.lstm = LSTMModel(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              num_layers=num_layers,
                              dropout=dropout).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_dim)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_dim)
        """

        # with torch.inference_mode():
        with torch.no_grad():  # disables gradients but keeps tensors usable
            x = self.autoencoder.Embedding(x)
        x = x.unsqueeze(1)  

        out = self.lstm(x)
        return out.squeeze(1)  # remove the sequence dimension

    def load_autoencoder(self,
                         address_autoencoder: str,
                         embedding_dim: int) -> None:
        """
        Load the autoencoder model from a file.
        Args:
            address_autoencoder (str): Path to the autoencoder model file
            embedding_dim (int): Dimension of the embedding space
        Returns:
            None: Loads the autoencoder model.
        """
        self.autoencoder = Autoencoder_CNN(embedding_dim=embedding_dim).to(self.device)
        self.autoencoder.eval()
        self.autoencoder.load_state_dict(torch.load(address_autoencoder, map_location=self.device))
        # self.autoencoder.requires_grad_(False)