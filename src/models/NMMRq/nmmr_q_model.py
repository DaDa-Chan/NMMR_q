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
        self.initialize_weights()
        
    def initialize_weights(self):
        # 遍历模型中的所有层
        for layer in self.layer_list:
            # 检查这一层是不是一个线性层 (nn.Linear)
            if isinstance(layer, nn.Linear):
            # 初始化权重 (Weight)
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            # 初始化偏置 (Bias)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
        
    def forward(self, x):
        
        for layer in self.layer_list[:-1]:
            x = torch.nn.functional.leaky_relu(layer(x))       
        x = self.layer_list[-1](x)
        x = torch.nn.functional.softplus(x)          
        return x
    