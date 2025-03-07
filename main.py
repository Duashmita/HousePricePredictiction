import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



file = pd.read_csv("housing_dataset.csv")

# Cleanes the data
print(file.info())
file = file.dropna(ignore_index=True)
file = file.drop_duplicates(ignore_index=True)
file['ocean_proximity'] = file['ocean_proximity'].str.upper()
file = pd.get_dummies(file, columns=['ocean_proximity'])
print(file.info())
file.to_csv("cleaned_houses.csv")
inputs = file.drop(columns="median_house_value")

num = file.to_numpy(dtype="float")
innum = inputs.to_numpy(dtype="float")

scaler = StandardScaler()
num = scaler.fit_transform(num)
innum = scaler.fit_transform(innum)

data = torch.tensor(num, dtype=torch.float)
inputse = torch.tensor(innum, dtype=torch.float)

#Training Data
train_inputs = inputse[:5000,:]
train_outputs = data[:5000,8]
#Test Data
test_inputs = inputse[5001,:].unsqueeze(0)
test_outputs = data[5001, 8].unsqueeze(0)
#### END DATASSET CODE ####

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
        # passing input through layers and activations and return it
        partial = self.layer1(input)
        partial = self.activation(partial)
        partial = self.layer2(partial)
        partial = self.activation(partial)
        output = self.layer3(partial)

        return output

# create the model, loss function, and optimizer
model = MyModel()
loss_function = nn.L1Loss() # Defining the mean squared equation loss function
optimizer = torch.optim.Adam(model.parameters(),lr=0.001) # Defing the learning step and choosing an optimiser for running

NUM_EPOCHS = 60
ls = []

# training loop
print("\nTRAINING:\n") 
for i in range(NUM_EPOCHS):
    pred = model(train_inputs)
    loss = loss_function(pred.squeeze(1), train_outputs)
    ls.append(loss.item())
    print(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# calculate testing loss
print("\nTESTING:\n")
with torch.no_grad():
    # make a prediction from the TEST inputs, and score it with the loss function
    pred = model(test_inputs)
    loss = loss_function(pred.squeeze(1), test_outputs) 
    ls.append(loss.item())
    print(loss.item())

torch.save(model.state_dict(),"Model.pt")

#Ploting 
plt.plot(ls, 'ro', label='MAE Error')
plt.scatter(len(ls)-1, ls[-1], color='blue', s=100, label='Testing Loss')
# plt.show()
