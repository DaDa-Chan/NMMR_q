import torch
import torch.nn as nn

class NMMR_Q_common(nn.Module):
    
    def __init__(self, input_dim, train_params):
        super().__init__()
        
        self.train_params = train_params
        self.network_width = train_params["network_width"]
        self.network_depth = train_params["network_depth"]

        self.layer_list = nn.ModuleList()
        for i in range(self.network_depth):
            if i == 0:
               self.layer_list.append(nn.Linear(input_dim, self.network_width)) 
            else:
                self.layer_list.append(nn.Linear(self.network_width, self.network_width))
        self.layer_list.append(nn.Linear(self.network_width, 1))
        
    def forward(self, x):
        
        for layer in self.layer_list[:-1]:
            x = nn.functional.leaky_relu(layer(x))
            
        x = self.layer_list[-1](x)
        x = nn.functional.softplus(x)
                    
        return x
    