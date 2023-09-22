import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

class TrainBetaDataset(Dataset):
    def __init__(self, seq_len, symbol, input_size):
        # dataset attributes definition
        self.seq_len = seq_len
        self.input_size = input_size
        self.symbol = symbol
        
        # load the csv 
        self.df = pd.read_csv("c:/users/Candidate/Data/beta_project_data.csv") 
        self.df.astype({col: np.float32 for col in self.df.columns[3:]})

        # adjust the close values to reflect adjustment factor
        self.df['Close'] = self.df['Close'].mul(self.df['Adjustment Factor'])

        # sort the csv by Date
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values(by="Date", ascending=True).reset_index(drop=True)
        
        # single out desired securities
        self.df = self.df[self.df['Symbol'] == symbol]
        
        # add percent return column
        self.df['Close_pct'] = self.df['Close'].pct_change()
        
        # add a 45 day rolling beta column
        self.df['Rolling Vol'] = self.df['Close_pct'].rolling(45).std() 
        
        # normalize the volume column
        self.df['Volume_pct'] = self.df['Volume'].pct_change()
        
        # make the training selection and scale it
        self.df = self.df.iloc[45:]
        self.df = self.df.reset_index(drop=True)
        
        scaler = MinMaxScaler()
        for col in ['Close_pct', 'Volume_pct', 'Rolling Vol']:
            self.df[col] = pd.DataFrame(scaler.fit_transform(pd.DataFrame(self.df[col])), columns=[col])
            
        self.df = self.df.iloc[: int(np.floor(len(self.df) * 0.8)) - 100]
        self.df = self.df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df) - self.seq_len - 45
    
    def __getitem__(self, idx):
        # initialize an empty tensor for our data
        data = torch.zeros(self.seq_len, self.input_size, dtype=torch.float32)
        vol = torch.zeros(1, dtype=torch.float32)
        
        # reference dictionary for our data. Do percent change for volume
        ref = {0:6, 1:7, 2:8}
        
        # fill in our empty data tensor
        for i in range(0, self.seq_len):
            for x in range(0, self.input_size):
                data[i][x] = torch.tensor(self.df.iloc[idx + i][ref[x]])
        
        # calculate target beta
        vol[0] = self.df.iloc[idx + self.seq_len + 45][7]
  
        return data, vol


# hyperparameters
input_size = 3 
train_batch_size = 80
test_batch_size = 80
seq_len = 100
stock = "T"
validation_split = 0.8

# dataset initilization
train_dataset = TrainBetaDataset(seq_len, stock, input_size)

# sampler intilizations
train_sampler = SequentialSampler(train_dataset)

# dataloader initializations
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, sampler=train_sampler)


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        # super class init call 
        super().__init__()
        
        # definition of hyperparameters
        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # layer definitions
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        
        # try simplifying fully connected, add dropout. For regression try sigmoid
        self.fc_1 = nn.Linear(hidden_size, 32)
        self.fc_2 = nn.Linear(32, num_classes)
        
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X):
        # hidden state initialization. Move state to init
        h0 = Variable(torch.randn(self.num_layers, X.size(0), self.hidden_size))
        c0 = Variable(torch.randn(self.num_layers, X.size(0), self.hidden_size))
        
        # propogation through layers
        out, (_, _) = self.lstm(X, (h0, c0))
        out = out[:, -1, :]

        out = self.sigmoid(out)
        out = self.dropout(out)
        out = self.fc_1(out)
        out = self.sigmoid(out)
        out = self.dropout(out)
        out = self.fc_2(out)

        return out

# more hyperparameters
hidden_size = 128
num_classes = 1
num_layers = 2
learning_rate = 0.001
num_epochs = 100

# model definition
model = LSTM(num_classes, input_size, hidden_size, num_layers)

loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')

def train(num_epochs, model, train_loader, loss_func, optimizer, scheduler):
#     total_steps = len(train_loader)
    
    for epoch in range(num_epochs):
        for _, (batch, beta) in enumerate(train_loader):
            # transfer data to CUDA cores
            if torch.cuda.is_available():
                batch, beta = batch.cuda(), beta.cuda()
            
            model.train()

            # store current output of the model
            output = model.forward(batch)
            optimizer.zero_grad()

            # calculate loss of current output/model
            train_loss = loss_func(output, beta)
            
            # model and lr optimization
            train_loss.backward()
            optimizer.step()
            
            model.eval()

        for _, (batch, beta) in enumerate(validation_loader):
            # transfer data to CUDA cores
            if torch.cuda.is_available():
                batch, beta = batch.cuda(), beta.cuda()
            
            # calculate current output of the model
            output = model.forward(batch)
            print(output, beta)
            # calculate current loss of 
            test_loss = loss_func(output, beta)
        
        # optimize the learning rate
        scheduler.step(test_loss)
        
        print(f"Epoch: {epoch + 1}; Train Loss: {train_loss.item():>4f}; Test Loss: {test_loss.item():>4f}")






