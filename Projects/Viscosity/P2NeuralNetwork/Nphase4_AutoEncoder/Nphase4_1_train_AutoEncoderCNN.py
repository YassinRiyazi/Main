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
from trainer.Base import train

# Set the random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set float32 matrix multiplication precision to medium
torch.set_float32_matmul_precision('high')

def prepare_data_tensor(dataloader, device):
    """Convert dataloader into a single tensor for the trainer"""
    all_data = []
    for data, _ in dataloader:
        all_data.append(data.to(device))
    return torch.cat(all_data, dim=0)

def train(
    data_dir='/media/d2u25/Dont/frames_Process_15_Patch',
    model_name='cnn_autoencoder',
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
    num_hard_samples=1000
):
    # Create directories
    os.makedirs(ckpt_save_path, exist_ok=True)
    
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
    model = networks.Autoencoder_CNN(embedding_dim=1000).to(device)
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
        ckpt_save_freq=ckpt_save_freq,
        ckpt_save_path=ckpt_save_path,
        ckpt_path=ckpt_path,
        report_path=report_path,
        lr_scheduler=scheduler,
        use_hard_negative_mining=use_hard_negative_mining,
        hard_mining_freq=hard_mining_freq,
        num_hard_samples=num_hard_samples
    )

if __name__ == '__main__':
    train_cnn_autoencoder() 