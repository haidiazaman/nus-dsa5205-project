import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self,num_features,num_classes):        
        super(FCN, self).__init__()
        # self.f = nn.Flatten()
        self.fcn = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
            nn.Dropout(0.5),          
            
            
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.5),          
            
            nn.Linear(256, num_classes),
        )


    def forward(self, x):
        # x = self.f(x)
        x = self.fcn(x)
        return x