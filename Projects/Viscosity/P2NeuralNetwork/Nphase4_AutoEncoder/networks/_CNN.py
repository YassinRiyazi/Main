import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple

class Encoder_CNN(nn.Module):
    """
    Encoder network for the autoencoder using CNN layers.
    This network compresses the input image into a lower-dimensional embedding.
    """
    def __init__(self,
                 embedding_dim: int = 100) -> None:
        """
        Initializes the encoder network.
        Args:
            embedding_dim (int): Dimension of the embedding space
        Returns:
            None: Initializes the encoder with convolutional layers followed by a fully connected layer.
        """
        super(Encoder_CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 1x200x200 → 16x100x100
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 32x50x50
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 64x25x25
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 128x13x13
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 13 * 13, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 200, 200)   
        Returns:
            torch.Tensor: Embedding tensor of shape (batch_size, embedding_dim)
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        return embedding

class Decoder_CNN(nn.Module):
    """
    Decoder network for the autoencoder using CNN layers.
    This network reconstructs the image from the lower-dimensional embedding.
    """
    def __init__(self,
                 embedding_dim: int = 100) -> None:
        """
        Initializes the decoder network.
        Args:
            embedding_dim (int): Dimension of the embedding space
        Returns:
            None: Initializes the decoder with a fully connected layer followed by transposed convolutional layers.
        """
        super(Decoder_CNN, self).__init__()
        self.fc = nn.Linear(embedding_dim, 128 * 13 * 13)
        self.deconv = nn.Sequential(
                                        nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # 64x25x25
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=0),  # 32x50x50
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=0), # 16x100x100
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=0),   # 1x200x200
                                        nn.Sigmoid()
                                    )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, embedding_dim)
        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, 1, 200, 200)
        """
        x = self.fc(x)
        x = x.view(x.size(0), 128, 13, 13)
        x = self.deconv(x)
        return x

class Autoencoder_CNN(nn.Module):
    def __init__(self,
                 embedding_dim: int = 100) -> None:
        """
        Initializes the autoencoder network.
        Args:
            embedding_dim (int): Dimension of the embedding space
        Returns:
            None: Initializes the encoder and decoder networks.
        """

        super(Autoencoder_CNN, self).__init__()
        # self.resizer = ResizeTo200()
        self.encoder = Encoder_CNN(embedding_dim)
        self.decoder = Decoder_CNN(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 200, 200)
        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, 1, 200, 200)
        """
        # x = self.resizer(x)
        embedding = self.encoder(x)
        recon = self.decoder(embedding)
        return recon#, embedding
