import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self):        
        super(FCN, self).__init__()
        
        self.fcn = nn.Sequential(
            nn.Linear(26, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.5),          
            
            nn.Linear(256, 3),
        )


    def forward(self, x):
        x = self.fcn(x)
        return x