import  os
import  torch
import  time
import  torch.nn    as      nn
import  pandas      as      pd
import  numpy       as      np
from    torch.optim import  lr_scheduler
from    tqdm        import  tqdm
from    datetime    import  datetime
from torchvision.utils import save_image
from typing import Callable, Optional, Union

def save_reconstructions(model: nn.Module,
                         dataloader: torch.utils.data.DataLoader,
                         device: torch.device,
                         save_dir: str,
                         epoch: int,
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
        for data, _ in dataloader:
            data = data.to(device)
            recon = model(data)
            # Take only the first num_samples
            originals = data[:num_samples]
            reconstructions = recon[:num_samples]
            # Save originals and reconstructions
            save_image(originals, os.path.join(save_dir, f'originals_epoch{epoch}.png'), nrow=num_samples)
            save_image(reconstructions, os.path.join(save_dir, f'reconstructions_epoch{epoch}.png'), nrow=num_samples)
            break  # Only process the first batch

class AverageMeter(object):
    """
    computes and stores the average and current value
    """
    def __init__(self, start_val=0, start_count=0, start_avg=0, start_sum=0):
        self.reset()
        self.val = start_val
        self.avg = start_avg
        self.sum = start_sum
        self.count = start_count

    def reset(self):
        """
        Initialize 'value', 'sum', 'count', and 'avg' with 0.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        """
        Update 'value', 'sum', 'count', and 'avg'.
        """
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def create_save_dir(base_path: str, model_name: str) -> os.PathLike:
    """
    Create a timestamped directory for saving model checkpoints and reports
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(base_path, f"{model_name}_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def save_model(file_path, file_name, model, optimizer=None):
    """
    Save model and optimizer state
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path, model, optimizer=None):
    """
    Load model and optimizer state from checkpoint
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer

def normal_accuracy(pred,labels):    
    return ((pred.argmax(dim=1)==labels).sum()/len(labels))*100

def teacher_forcing_decay(epoch, num_epochs):
    initial_tf_ratio = 1.0
    final_tf_ratio = 0.01
    decay_rate = (final_tf_ratio / initial_tf_ratio) ** (1 / (num_epochs - 1))

    tf_ratio = max(0.01,initial_tf_ratio * (decay_rate ** epoch))
    return tf_ratio

def hard_negative_mining(model, dataloader, criterion, device, num_hard_samples=2000):
    """
    Select the hardest examples (highest loss) from the dataset
    Returns a new DataLoader containing only the hard examples
    """
    model.eval()
    losses = []
    all_data = []
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            recon = model(data)
            loss = criterion(recon, data)
            # If loss is a scalar, reshape it to match batch size
            if loss.dim() == 0:
                loss = loss.unsqueeze(0)
            # Calculate per-sample loss
            per_sample_loss = loss.view(-1)
            losses.extend(per_sample_loss.cpu().numpy())
            all_data.append(data.cpu().numpy())
    
    # Convert to numpy arrays
    losses = np.array(losses)
    all_data = np.concatenate(all_data, axis=0)
    
    # Get indices of hardest examples
    hard_indices = np.argsort(losses)[-num_hard_samples:]
    
    # Create new dataset with hard examples
    hard_data = all_data[hard_indices]
    hard_dataset = torch.utils.data.TensorDataset(torch.from_numpy(hard_data), torch.zeros(len(hard_data)))
    
    # Create new dataloader
    hard_loader = torch.utils.data.DataLoader(
        hard_dataset,
        batch_size=min(128, num_hard_samples),  # Use smaller batch size for hard examples
        shuffle=True,
        num_workers=4
    )
    
    return hard_loader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def handler_selfSupervised(Args:tuple[torch.Tensor, torch.Tensor],
                           criterion: nn.Module,
                           model: nn.Module):
    data = Args[0].data.to(device)
    data = data.contiguous()

    output = model(data)
    loss = criterion(output, data)
    return output, loss



def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: nn.Module,
    epochs: int,
    device: str,
    model_name: str,
    ckpt_save_freq: int,
    ckpt_save_path: Union[str, os.PathLike],

    handler: Callable[[tuple[torch.Tensor, torch.Tensor], nn.Module, nn.Module], None],
    handler_postfix: Optional[Callable] = save_reconstructions,

    ckpt_path: Union[str, os.PathLike] = None,
    report_path: Union[str, os.PathLike] = None,
    lr_scheduler = None,
    Validation_save_threshold: float = 0.0,
    use_hard_negative_mining: bool = True,
    hard_mining_freq: int = 1,  # Perform hard negative mining every N epochs
    num_hard_samples: int = 1000  # Number of hard examples to select
) -> tuple[nn.Module, nn.Module, pd.DataFrame]:
    """
    Standard training loop for autoencoder models with hard negative mining
    
    Args:
        model (nn.Module): PyTorch model
        train_loader (torch.utils.data.DataLoader): DataLoader for training data
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data
        criterion (nn.Module): Loss function
        optimizer (nn.Module): Optimizer
        epochs (int): Number of training epochs
        device (str): Device to train on ('cuda' or 'cpu')
        model_name (str): Name of the model for saving checkpoints
        ckpt_save_freq (int): Frequency of checkpoint saving (in epochs)
        ckpt_save_path (Union[str, os.PathLike]): Path to save checkpoints
        ckpt_path (Union[str, os.PathLike]): Path to load checkpoint from (if resuming training)
        report_path (Union[str, os.PathLike]): Path to save training report
        lr_scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        Validation_save_threshold (float): Threshold for saving best validation model
        use_hard_negative_mining (bool): Whether to use hard negative mining
        hard_mining_freq (int): Frequency of hard negative mining (in epochs)
        num_hard_samples (int): Number of hard examples to select

    Returns:
        model (nn.Module): Trained model
        optimizer (nn.Module): Optimizer with updated state
        report (pd.DataFrame): Training report with metrics

    TODO:
        - Plot training loss over epochs real time in the terminal or a window
    """
    # Create timestamped directory for this training run
    save_dir = create_save_dir(ckpt_save_path, model_name)
    print(f"Saving checkpoints and reports to: {save_dir}")
    
    model = model.to(device)
    
    if ckpt_path is not None:
        model, optimizer = load_model(ckpt_path=ckpt_path, model=model, optimizer=optimizer)

    # Initialize training report
    report = pd.DataFrame(columns=[
        "model_name",
        "mode",
        "epoch",
        "learning_rate",
        "batch_size",
        "batch_index",
        "loss_batch",
        "avg_train_loss_till_current_batch",
        "avg_val_loss_till_current_batch",
    ])
    
    # Set explicit dtypes for numeric columns
    numeric_columns = ["epoch", "learning_rate", "batch_size", "batch_index", 
                      "loss_batch", "avg_train_loss_till_current_batch", 
                      "avg_val_loss_till_current_batch"]
    for col in numeric_columns:
        report[col] = report[col].astype(float)
    
    best_val_loss = float('inf')
    
    
    for epoch in tqdm(range(1, epochs + 1)):
                # Perform hard negative mining if enabled and it's time
        if (use_hard_negative_mining and epoch % hard_mining_freq == 0) and epoch > 4:
            print(f"Performing hard negative mining at epoch {epoch}")
            current_train_loader = hard_negative_mining(
                model, train_loader, criterion, device, num_hard_samples
            )
        else:
            current_train_loader = train_loader
        
        
        model.train()
        loss_avg_train = AverageMeter()
        
        train_loop = tqdm(current_train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch_idx, Args in enumerate(train_loop):
            optimizer.zero_grad()
            
            output, loss = handler(Args, criterion, model)

            # Backward pass
            loss.contiguous()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            loss_avg_train.update(loss.item(), Args[0].size(0))
            
            # Update progress bar
            train_loop.set_postfix(loss=loss_avg_train.avg, lr=optimizer.param_groups[0]["lr"])
            
            # Create new row with explicit values
            new_row = {
                "model_name": model_name,
                "mode": "train",
                "epoch": float(epoch),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "batch_size": float(Args[0].size(0)),
                "batch_index": float(batch_idx),
                "loss_batch": float(loss.item()),
                "avg_train_loss_till_current_batch": float(loss_avg_train.avg),
                "avg_val_loss_till_current_batch": np.nan
            }
            
            # Append to report
            report = pd.concat([report, pd.DataFrame([new_row])], ignore_index=True)

        # Validation phase
        model.eval()
        loss_avg_val = AverageMeter()
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]")
            for batch_idx, (data, _) in enumerate(val_loop):
                data = data.to(device)
                
                output, loss = handler(Args, criterion, model)
                
                # Update metrics
                loss_avg_val.update(loss.item(), Args[0].size(0))
                
                # Update progress bar
                val_loop.set_postfix(loss=loss_avg_val.avg)
                
                # Create new row with explicit values
                new_row = {
                    "model_name": model_name,
                    "mode": "val",
                    "epoch": float(epoch),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "batch_size": float(Args[0].size(0)),
                    "batch_index": float(batch_idx),
                    "loss_batch": float(loss.item()),
                    "avg_train_loss_till_current_batch": np.nan,
                    "avg_val_loss_till_current_batch": float(loss_avg_val.avg)
                }
                
                # Append to report
                report = pd.concat([report, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save sample reconstructions from validation set
        if handler_postfix is not None:
            handler_postfix(
                model=model,
                dataloader=val_loader,
                device=device,
                save_dir=os.path.join(save_dir, f"reconstructions_epoch{epoch}"),
                epoch=epoch,
                num_samples=8
            )
        
        # Save checkpoint
        if epoch % ckpt_save_freq == 0:
            save_model(
                file_path=save_dir,
                file_name=f"ckpt_{model_name}_epoch{epoch}.ckpt",
                model=model,
                optimizer=optimizer,
            )
        
        # Save best model based on validation loss
        if loss_avg_val.avg < best_val_loss:
            best_val_loss = loss_avg_val.avg
            save_model(
                file_path=save_dir,
                file_name=f"best_{model_name}.ckpt",
                model=model,
                optimizer=optimizer,
            )
        
        # Update learning rate
        if (lr_scheduler is not None) and (epoch % hard_mining_freq == 3):
            lr_scheduler.step()
        
        # Save report
        if report_path is not None:
            report.to_csv(os.path.join(save_dir, f"{model_name}_report.csv"), index=False)
    
    # Save final model state as .pt file
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_final.pt"))
    
    return model, optimizer, report