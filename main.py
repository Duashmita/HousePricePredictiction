import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

### DATA CLEANING, BATCHING AND SCALING ###
# Data Class
class DataLoaders(Dataset):

    def __init__(self,file):
        
        # Data cleaning
        file = file.dropna(ignore_index=True)
        file = file.drop_duplicates(ignore_index=True)
        file['ocean_proximity'] = file['ocean_proximity'].str.upper()
        file = pd.get_dummies(file, columns=['ocean_proximity'])
        new_order = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    'ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND', 'ocean_proximity_ISLAND',
    'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN',
    'median_house_value']
        file = file[new_order]

        # Converting data types
        self.data = file.to_numpy(dtype="float")
        self.data = torch.tensor(self.data, dtype=torch.float)

        # defining output
        self.outputs = self.data[:,13]
        self.inputs = self.data[:,0:13]


    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

    def __len__(self):
        return len(self.data)

file = pd.read_csv("housing_dataset.csv")
dataObject = DataLoaders(file)

#### AI MODEL ####

# Class for model defination
class MyModel(nn.Module):

    def __init__(self):
        #initialising layers and activation
        super().__init__()
        self.layer1 = nn.Linear(13,32)
        self.layer2 = nn.Linear(32,10)
        self.layer3 = nn.Linear(10,1)
        self.activation = nn.ReLU()

    def forward(self, input):
        partial = self.layer1(input)
        partial = self.activation(partial)
        partial = self.layer2(partial)
        partial = self.activation(partial)
        output = self.layer3(partial)
        return output

# Hyperparameters
model = MyModel()
loss_function = nn.L1Loss() # Defining the mean squared equation loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Defining the learning step and choosing an optimiser for running
NUM_EPOCHS = 20
loss_train = []
loss_test = []
 
# Batching data for loops
Dataloader = DataLoader(dataObject, batch_size=45, shuffle=True)

# TRAINING LOOP
for i in range(NUM_EPOCHS+1):
    for i in Dataloader:
        pred = model(i[0]).squeeze(1)
        loss = loss_function(pred, i[1])
        loss_train.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# TESTING LOOP
for i in Dataloader:
    if random.choice([True, False]):
        pred = model(i[0]).squeeze(1)
        loss = loss_function(pred, i[1])
        loss_test.append(loss.item())

torch.save(model.state_dict(), "Model.pt")