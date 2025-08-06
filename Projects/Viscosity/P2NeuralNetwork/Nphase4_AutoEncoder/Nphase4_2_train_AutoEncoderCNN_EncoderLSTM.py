"""
    Author:         Yassin Riyazi
    Date:           04-08-2025
    Description:    Train an embedding-based LSTM for time series data.

    TODO:
        - 

"""

import  torch
import  dataset
import  torch
import  numpy               as      np
import  torch.nn            as      nn
import  torch.optim         as      optim
from    torch.utils.data    import  DataLoader
import networks
import dataset as DSS
import time

batch_size  = 40
data_dir='/media/d2u25/Dont/frames_Process_15_Patch'
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



model = networks.AutoEncoder_CNN_LSTM.Encoder_LSTM(
    address_autoencoder= 'Projects/Viscosity/P2NeuralNetwork/Nphase4_AutoEncoder/checkpoints/cnn_autoencoder_20250804_130532/cnn_autoencoder_final.pt',
    input_dim=1000,  # Adjust based on your data
    hidden_dim=128,  # Adjust based on your model architecture
    num_layers=2,  # Number of LSTM layers
    dropout=0.2,  # Dropout rate
)

# model = LSTMModel(input_dim, hidden_dim, num_layers, dropout)
# Define the loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()



# Train the model
for epoch in range(100):
    for batch in train_loader:
        inputs, labels = batch
        inputs = inputs.to(model.device)  # Move inputs to the device
        # labels = labels.to(model.device)  # Move labels to the device

        optimizer.zero_grad()
        outputs = model(inputs)  # inputs is a tensor of shape [batch, numberOfSample, 1000]
        loss = criterion(outputs, torch.rand(size=[batch_size],device=model.device))  # labels is a tensor of shape [batch, 1]
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        break
            
    break