import torch 
import torch.nn as nn
import torch.nn.functional as f

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1layer = nn.Conv2d(3,32,3,padding=3)
        self.pool = nn.MaxPool2d(2,2)
        self.linear = None
    
    def forward(self,x):
        x = self.conv1layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        
        if self.linear is None:
            flattened_size = x.view(x.size(0), -1).size(1)
            self.linear = nn.Linear(flattened_size, 10)

        x = self.linear(x)
        x = f.softmax(x, dim=1)
        return x
