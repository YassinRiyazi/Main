"""
    Author: Yassin Riyazi
    Date: 04-08-2025
    Description: Train a CNN-based autoencoder for image data.

"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
import dataset as DSS
import networks
from torchvision.utils import save_image
from typing import Callable, Optional, Union

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../', 'src/PyThon/NeuralNetwork/trainer')))
from Base import train

# Set the random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set float32 matrix multiplication precision to medium
torch.set_float32_matmul_precision('high')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EmbeddingSize = 256



def handler_selfSupervised_dataHandler(Args: tuple[torch.Tensor, torch.Tensor],
                                       model: nn.Module,
                                       device: torch.device) -> torch.Tensor:
    data = Args[0].squeeze(1).to(device)
    return data, model(data)

def handler_selfSupervised_loss(criterion, output, data):
    return criterion(output, data)


def handler_selfSupervised(Args:tuple[torch.Tensor, torch.Tensor],
                           criterion: nn.Module,
                           model: nn.Module,
                           device: torch.device = 'cuda') -> tuple[torch.Tensor, torch.Tensor]:
    data, output    = handler_selfSupervised_dataHandler(Args, model, device)
    loss            = handler_selfSupervised_loss(criterion, output, data)
    return output, loss


def save_reconstructions(
                         model: nn.Module,
                         dataloader: torch.utils.data.DataLoader,
                         device: torch.device,
                         save_dir: str,
                         epoch: int,
                         dataHandler: Callable = handler_selfSupervised_dataHandler,
                         num_samples: int = 8) -> None:
    """Save a batch of original and reconstructed images from the dataloader.
    Args:
        model (nn.Module): The trained autoencoder model
        dataloader (torch.utils.data.DataLoader): DataLoader for validation or test set
        device (torch.device): Device to run the model on
        save_dir (str): Directory to save the images
        epoch (int): Current epoch number for naming
        num_samples (int): Number of samples to save from the batch
    Returns:
        None: Saves images to the specified directory.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i, Args in enumerate(dataloader):
            _, recon = dataHandler(Args, model, device)
            # Take only the first num_samples
            originals = Args[0][:num_samples]

            total_samples = originals.size(0)
            rand_indices = torch.randperm(total_samples)[:num_samples]
            originals = originals[rand_indices] 
            originals = originals.squeeze(1)  # Remove channel dimension if present

            reconstructions = recon[:num_samples]
            # Save originals and reconstructions
            save_image(originals, os.path.join(save_dir, f'originals_epoch{epoch}_batch{i}.png'), nrow=num_samples)
            save_image(reconstructions, os.path.join(save_dir, f'reconstructions_epoch{epoch}_batch{i}.png'), nrow=num_samples)
            if i >= 5:  # Limit to first 5 batches
                break  # Only process the 5 batch


def trainer(
    data_dir='/media/d2u25/Dont/frames_Process_15_Patch',
    model_name=f'cnn_autoencoder_{EmbeddingSize}',
    epochs=20,
    batch_size=128,
    learning_rate=0.0005,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    ckpt_save_freq=3,
    ckpt_save_path=os.path.join(os.path.dirname(__file__), 'checkpoints'),
    ckpt_path=None,
    report_path=os.path.join(os.path.dirname(__file__), 'training_report.csv'),
    use_hard_negative_mining=False,
    hard_mining_freq=2,
    num_hard_samples=1000,
    EmbeddingSize=EmbeddingSize
):
    # Create directories
    # os.makedirs(ckpt_save_path, exist_ok=True)
    
    # Load dataset
    dataset = DSS.loc_ImageDataset(
                                        data_dir=data_dir,
                                        skip=4,
                                        load_from_file=True,
                                        use_yaml=False
                                    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Optimize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16, shuffle=False, pin_memory=True)
    
    # Initialize model and optimizer
    model = networks.Autoencoder_CNN(embedding_dim=EmbeddingSize).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
       
    # Learning rate scheduler
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)  # Divide by 5 every epoch 0.2

    # Train the model
    model, optimizer, report = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        model_name=model_name,
        handler=handler_selfSupervised,
        handler_postfix=save_reconstructions,
        ckpt_save_freq=ckpt_save_freq,
        ckpt_save_path=ckpt_save_path,
        ckpt_path=ckpt_path,
        report_path=report_path,
        lr_scheduler=scheduler,
        use_hard_negative_mining=use_hard_negative_mining,
        hard_mining_freq=hard_mining_freq,
        num_hard_samples=num_hard_samples
    )


def Main(EmbeddingSize: int = 256):
    """
    Main function to run the training process.
    Args:
        EmbeddingSize (int): Size of the embedding dimension for the autoencoder.
    """
    trainer(
        data_dir='/media/d2u25/Dont/frames_Process_15_Patch',
        model_name=f'cnn_autoencoder_{EmbeddingSize}',
        epochs=10,
        batch_size=128,
        learning_rate=0.0005,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        ckpt_save_freq=3,
        ckpt_save_path=os.path.join(os.path.dirname(__file__), 'checkpoints'),
        ckpt_path=None,
        report_path=os.path.join(os.path.dirname(__file__), 'training_report.csv'),
        use_hard_negative_mining=False,
        hard_mining_freq=2,
        num_hard_samples=1000,
        EmbeddingSize=EmbeddingSize)
    
if __name__ == '__main__':
    for ii in [10, 50, 256, 512, 1024]:
        Main(EmbeddingSize=ii)