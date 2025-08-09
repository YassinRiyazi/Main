"""
    Author:         Yassin Riyazi
    Date:           04-08-2025
    Description:    Train an embedding-based LSTM for time series data.

    TODO:
        - 

"""
import  os
import  time
import  glob 
import  torch
import  dataset
import  networks
import  numpy               as      np
import  dataset             as      DSS
import  torch.nn            as      nn
import  torch.optim         as      optim
from    torch.utils.data    import  DataLoader
from    typing              import  Callable, Optional, Union

import  sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../', 'src/PyThon/NeuralNetwork/trainer')))
from Base import train

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
    loss = criterion(output, Args[1].view(-1))
    return output, loss

def save_reconstructions(
                         model: nn.Module,
                         dataloader: torch.utils.data.DataLoader,
                         device: torch.device,
                         save_dir: str,
                         epoch: int,
                         num_samples: int = 8) -> None:
    """Save a batch of original and reconstructed images from the dataloader and save target/predicted values to a text file.
    Args:
        model (nn.Module): The trained autoencoder model
        dataloader (torch.utils.data.DataLoader): DataLoader for validation or test set
        device (torch.device): Device to run the model on
        save_dir (str): Directory to save the images and text file
        epoch (int): Current epoch number for naming
        num_samples (int): Number of samples to save from the batch
    Returns:
        None: Saves images and text file to the specified directory.
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        for i, Args in enumerate(dataloader):
            Args = [arg.contiguous().to(device) for arg in Args]
            model.lstm.reset_states(Args[0])  # Reset LSTM states before processing a new batch
            output = model(Args[0])

            target = Args[1].view(-1)
            predicted = output.view(-1)

            # Save target and predicted values to a text file
            text_file_path = os.path.join(save_dir, f"reconstructions_epoch_{epoch}_batch_{i}.txt")
            with open(text_file_path, 'w') as f:
                f.write("Sample Index\tTarget Value\tPredicted Value\n")
                for j in range(min(num_samples, len(target))):
                    f.write(f"{j}\t{target[j].item():.6f}\t{predicted[j].item():.6f}\n")

            if i >= 5:  # Limit to first 5 batches
                break  # Only process the 5 batch


def train_lstm_model(CnnAutoEncoderEmbdSize = 256,
                     SEQUENCE_LENGTH = 20,
                     hidden_dim=256,
                     epochs = 5) -> None:
    """
    This function is a placeholder for training the LSTM model.
    It can be extended to include specific logic for training tasks.
    """
    batch_size  = 100
    data_dir    = '/media/d2u25/Dont/frames_Process_15_Patch'

    # Load dataset
    dataset = DSS.loc_ImageDataset(
                                    data_dir=data_dir,
                                    skip=4,
                                    sequence_length=SEQUENCE_LENGTH,
                                    load_from_file=True,
                                    use_yaml=False
                                )

    train_size                      = int(0.8 * len(dataset))
    val_size                        = len(dataset) - train_size
    train_dataset, val_dataset      = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Optimize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, num_workers=16, shuffle=False, pin_memory=True)


    
    model = networks.AutoEncoder_CNN_LSTM.Encoder_LSTM(
        address_autoencoder= glob.glob(f'Projects/Viscosity/P2NeuralNetwork/Nphase4_AutoEncoder/checkpoints/cnn_autoencoder_{CnnAutoEncoderEmbdSize}*/*.pt')[0],
        input_dim=CnnAutoEncoderEmbdSize,  # Adjust based on your data
        hidden_dim=hidden_dim,  # Adjust based on your model architecture
        num_layers=2,  # Number of LSTM layers
        dropout=0.1,  # Dropout rate
        sequence_length=SEQUENCE_LENGTH,
    )

    # model = LSTMModel(input_dim, hidden_dim, num_layers, dropout)
    # Define the loss function and optimizer
    optimizer = optim.SGD(model.parameters(), lr=1e-2,)
    criterion = nn.MSELoss()

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        criterion = criterion,
        optimizer = optimizer,
        epochs = epochs,
        device = device,
        model_name = f"Encoder_{CnnAutoEncoderEmbdSize}_LSTM_HD{hidden_dim}_SL{SEQUENCE_LENGTH}",

        handler = handler_supervised,
        handler_postfix=save_reconstructions,

        ckpt_save_freq=3,
        ckpt_save_path=os.path.join(os.path.dirname(__file__), 'checkpoints'),
        ckpt_path=None,
        report_path=os.path.join(os.path.dirname(__file__), 'training_report.csv'),
        use_hard_negative_mining=False,

        lr_scheduler = lr_scheduler
    )


if __name__ == "__main__":
    for CnnAutoEncoderEmbdSize in reversed([10, 50, 128, 256, 512, 1024]):
        for SEQUENCE_LENGTH in [1, 10]:
            for hidden_dim in reversed([128, 256]):
                print(f"Training with CnnAutoEncoderEmbdSize={CnnAutoEncoderEmbdSize}, SEQUENCE_LENGTH={SEQUENCE_LENGTH}, hidden_dim={hidden_dim}")
                train_lstm_model(CnnAutoEncoderEmbdSize=CnnAutoEncoderEmbdSize,
                                 SEQUENCE_LENGTH=SEQUENCE_LENGTH,
                                 hidden_dim=hidden_dim,
                                 epochs=30)    
    