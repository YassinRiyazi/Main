import  os
import  torch
import  time
import  torch.nn    as      nn
import  pandas      as      pd
import  numpy       as      np
from    torch.optim import  lr_scheduler
from    tqdm        import  tqdm
from    datetime    import  datetime
from typing import Callable, Optional, Union

import  subprocess

def monitor_gpu_temperature(threshold: int = 70,
                            sleep_seconds: int = 5,
                            gpu_id: int = 0,
                            verbose: bool = False) -> None:
    """
    Checks the GPU temperature and sleeps if it exceeds a threshold.

    Args:
        threshold (int): Temperature in Celsius above which the function sleeps.
        sleep_seconds (int): Number of seconds to sleep when the threshold is exceeded.
        gpu_id (int): ID of the GPU to monitor.

    returns:
        None: The function will print a warning and sleep if the temperature exceeds the threshold.
    """
    try:
        # Query GPU temperature using nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu=temperature.gpu", "--format=csv,noheader,nounits", f"-i={gpu_id}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        temp = int(result.stdout.strip())

        if temp > threshold:
            if verbose:
                print(f"[WARNING] GPU {gpu_id} temperature {temp}°C exceeds {threshold}°C. Sleeping for {sleep_seconds}s...")
            time.sleep(sleep_seconds)
        else:
            if verbose:
                print(f"[INFO] GPU {gpu_id} temperature {temp}°C is within safe limits.")

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to get GPU temperature: {e.stderr}")
    except ValueError:
        print("[ERROR] Could not parse GPU temperature.")

class AverageMeter(object):
    """
    computes and stores the average and current value

    Author: 
        - Farshad Sangari
        
    Date: 08-08-2023
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


def save_model(file_path: str,
               file_name: str,
               model: nn.Module,
               optimizer=None) -> None:
    """
    Save model and optimizer state

    Args:
        file_path (str): Directory to save the model
        file_name (str): Name of the file to save the model
        model (nn.Module): PyTorch model to save
        optimizer (Optional[nn.Module]): Optimizer to save (if available)
    Returns:
        None: Saves the model state to the specified file

    Authors: 
        - Yassin Riyazi
        - Farshad Sangari

    Date: 08-08-2025
    """
    state_dict = dict()
    state_dict["model"] = model.state_dict()

    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    torch.save(state_dict, os.path.join(file_path, file_name))


def load_model(ckpt_path: Union[str, os.PathLike],
               model: nn.Module,
               optimizer=None)-> tuple[nn.Module, Optional[nn.Module]]:
    """
    Load model and optimizer state from checkpoint
    Args:
        ckpt_path (Union[str, os.PathLike]): Path to the checkpoint file
        model (nn.Module): PyTorch model to load state into
        optimizer (Optional[nn.Module]): Optimizer to load state into (if available)
    Returns:
        model (nn.Module): Model with loaded state
        optimizer (Optional[nn.Module]): Optimizer with loaded state (if provided)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer

def normal_accuracy(pred: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Calculate the accuracy of predictions against true labels.
    Args:
        pred (torch.Tensor): Predictions from the model
        labels (torch.Tensor): True labels
    Returns:
        float: Accuracy as a percentage
    """
    return ((pred.argmax(dim=1) == labels).sum() / len(labels)) * 100

def teacher_forcing_decay(epoch: int, num_epochs: int) -> float:
    """
    Calculate the teacher forcing ratio for a given epoch.
    Args:
        epoch (int): Current epoch number
        num_epochs (int): Total number of epochs
    Returns:
        float: Teacher forcing ratio for the current epoch"""
    initial_tf_ratio = 1.0
    final_tf_ratio = 0.01
    decay_rate = (final_tf_ratio / initial_tf_ratio) ** (1 / (num_epochs - 1))

    tf_ratio = max(0.01, initial_tf_ratio * (decay_rate ** epoch))
    return tf_ratio

def HardNegativeMiningPostHandler(args: tuple[torch.Tensor, ...]) -> np.ndarray:
    """
    Post-processing handler for hard negative mining.
    This function can be customized to save or visualize hard negative samples.
    Currently, it does nothing but can be extended as needed.
    Args:
        args (tuple[torch.Tensor, ...]): Tuple containing the data and possibly other tensors
    Returns:
        np.ndarray: Processed data, currently just returns the first tensor in args as a numpy array
    """
    return args[0].numpy()  # Assuming args is a tuple with the first element being the data

def hard_negative_mining(model: nn.Module,
                         dataloader: torch.utils.data.DataLoader,
                         handler: Callable, #TODO: Make this more flexible for different model types
                         HardNegativeMiningPostHandler: Callable,
                         criterion: nn.Module,
                         device: str,
                         num_hard_samples: int = 2000) -> torch.utils.data.DataLoader:
    """
    Select the hardest examples (highest loss) from the dataset
    Returns a new DataLoader containing only the hard examples

    Args:
        model (nn.Module): The trained model to evaluate
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset
        criterion (nn.Module): Loss function to compute the loss
        device (str): Device to run the model on ('cuda' or 'cpu')
        num_hard_samples (int): Number of hard examples to select
    Returns:
        torch.utils.data.DataLoader: DataLoader containing only the hard examples

    TODO:
        - Add handler for different model types (e.g., CNN, LSTM)
    """
    model.eval()
    losses = []
    all_data = []
    
    with torch.no_grad():
        for args in dataloader:
            output, loss = handler(args, criterion, model)
            # If loss is a scalar, reshape it to match batch size
            if loss.dim() == 0:
                loss = loss.unsqueeze(0)
            # Calculate per-sample loss
            per_sample_loss = loss.view(-1)
            losses.extend(per_sample_loss.cpu().numpy())
            all_data.append(HardNegativeMiningPostHandler(args))
    
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
    handler_postfix: Union[Callable, None],

    ckpt_path: Union[str, os.PathLike] = None,
    report_path: Union[str, os.PathLike] = None,
    lr_scheduler = None,
    Validation_save_threshold: float = 0.0,
    use_hard_negative_mining: bool = True,
    hard_mining_freq: int = 1,  # Perform hard negative mining every N epochs
    num_hard_samples: int = 1000,  # Number of hard examples to select
    GPU_temperature: int = 70,
    GPU_overheat_sleep: float = 5.0,
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
        GPU_temperature (int): Temperature threshold for GPU monitoring
        GPU_overheat_sleep (float): Sleep time in seconds if GPU temperature exceeds threshold

    Returns:
        model (nn.Module): Trained model
        optimizer (nn.Module): Optimizer with updated state
        report (pd.DataFrame): Training report with metrics

    TODO:
        - Plot training loss over epochs real time in the terminal or a window

    Authors: 
        - Yassin Riyazi
        - Farshad Sangari
        
    Date: 08-08-2025
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
                "avg_val_loss_till_current_batch": np.nan,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            
            # Append to report
            report = pd.concat([report, pd.DataFrame([new_row])], ignore_index=True)
            if batch_idx % 10 == 0:  # Monitor GPU temperature every 10 batches
                monitor_gpu_temperature(threshold=GPU_temperature, sleep_seconds=GPU_overheat_sleep)

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
                    "avg_val_loss_till_current_batch": float(loss_avg_val.avg),
                    
                }
                
                # Append to report
                report = pd.concat([report, pd.DataFrame([new_row])], ignore_index=True)
                if batch_idx % 10 == 0:  # Monitor GPU temperature every 10 batches
                    monitor_gpu_temperature(threshold=GPU_temperature, sleep_seconds=GPU_overheat_sleep)
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
        if (lr_scheduler is not None):
            lr_scheduler.step()
        
        # Save report
        if report_path is not None:
            report.to_csv(os.path.join(save_dir, f"{model_name}_report.csv"), index=False)
    
    # Save final model state as .pt file
    torch.save(model.state_dict(), os.path.join(save_dir, f"{model_name}_final.pt"))
    
    return model, optimizer, report