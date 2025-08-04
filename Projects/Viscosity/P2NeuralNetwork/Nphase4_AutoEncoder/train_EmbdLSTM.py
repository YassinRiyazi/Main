
import  os
import  torch
import  numpy               as      np
from    torch.utils.data    import  Dataset
from    torch.utils.data import DataLoader

import _dataset

import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # output layer

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # take the last hidden state
        out = self.fc(out)
        return out

# Define the model
input_dim = 1000  # input dimension
hidden_dim = 128  # hidden dimension
num_layers = 2  # number of LSTM layers
dropout = 0.2  # dropout rate

batch_size= 40


dataset = _dataset.TimeSeriesDataset("Data")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Optimize DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=16, shuffle=False, pin_memory=True)

model = LSTMModel(input_dim, hidden_dim, num_layers, dropout)
# Define the loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()



# Train the model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(torch.rand(size=[10,30,1000]))  # inputs is a tensor of shape [batch, numberOfSample, 1000]
    loss = criterion(outputs, torch.rand(size=[10]))  # labels is a tensor of shape [batch, 1]
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')