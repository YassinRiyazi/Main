"""
    Author:         Yassin Riyazi
    Date:           04-08-2025
    Description:    Train an embedding-based LSTM for time series data.

    TODO:
        - 

"""
import os
import  torch
import  dataset
import  numpy               as      np
import  torch.nn            as      nn
import  torch.optim         as      optim
from    torch.utils.data    import  DataLoader
import networks
import dataset as DSS
import time
import trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set float32 matrix multiplication precision to medium
torch.set_float32_matmul_precision('high')



def handler_supervised(Args:tuple[torch.Tensor, torch.Tensor],
                       criterion: nn.Module,
                       model: nn.Module):
    """
    This function is a placeholder for handling supervised training.
    It can be extended to include specific logic for supervised learning tasks.
    """
    Args = [arg.contiguous().to(device) for arg in Args]
    model.lstm.reset_states(Args[0])  # Reset LSTM states before processing a new batch
    output = model(Args[0])
    loss = criterion(output, Args[1])
    return output, loss


SEQUENCE_LENGTH = 20
batch_size  = 100
data_dir='/media/d2u25/Dont/frames_Process_15_Patch'
# Load dataset
dataset = DSS.loc_ImageDataset(
                                    data_dir=data_dir,
                                    skip=4,
                                    sequence_length=SEQUENCE_LENGTH,
                                    load_from_file=True,
                                    use_yaml=False
                                )

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Optimize DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16, shuffle=False, pin_memory=True)



model = networks.AutoEncoder_CNN_LSTM.Encoder_LSTM(
    address_autoencoder= 'Projects/Viscosity/P2NeuralNetwork/Nphase4_AutoEncoder/checkpoints/cnn_autoencoder_20250804_130532/cnn_autoencoder_final.pt',
    input_dim=1000,  # Adjust based on your data
    hidden_dim=256,  # Adjust based on your model architecture
    num_layers=2,  # Number of LSTM layers
    dropout=0.2,  # Dropout rate
    sequence_length=SEQUENCE_LENGTH,
)

# model = LSTMModel(input_dim, hidden_dim, num_layers, dropout)
# Define the loss function and optimizer
optimizer = optim.SGD(model.parameters(), lr=1e-4,)
criterion = nn.MSELoss()

trainer.train(
    model = model,
    train_loader = train_loader,
    val_loader = val_loader,
    criterion = criterion,
    optimizer = optimizer,
    epochs = 100,
    device = device,
    model_name = "Encoder_LSTM",

    handler = handler_supervised,
    handler_postfix = None,

    ckpt_save_freq=3,
    ckpt_save_path=os.path.join(os.path.dirname(__file__), 'checkpoints'),
    ckpt_path=None,
    report_path=os.path.join(os.path.dirname(__file__), 'training_report.csv'),
    use_hard_negative_mining=False,)