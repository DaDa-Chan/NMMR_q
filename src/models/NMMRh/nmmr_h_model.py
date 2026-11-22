import torch
import torch.nn as nn

from src.models.experiment_common import read_output_clip


class MLP_for_NMMR(nn.Module):
    def __init__(self, input_dim, train_params):
        super(MLP_for_NMMR, self).__init__()
        self.train_params = train_params
        self.network_width = train_params['network_width']
        self.network_depth = train_params['network_depth']
        self.layer_list = nn.ModuleList()
        for i in range(self.network_depth):
            if i == 0:
                self.layer_list.append(nn.Linear(input_dim, self.network_width))
            else:
                self.layer_list.append(nn.Linear(self.network_width, self.network_width))
        self.layer_list.append(nn.Linear(self.network_width, 1))
        # Optional symmetric clip B, so h lies in [-B, B]. None (the default,
        # also for configs without the key) leaves the output unchanged.
        self.output_clip = read_output_clip(train_params, "h_output_clip")

    def forward(self, x):
        for ix, layer in enumerate(self.layer_list):
            if ix == self.network_depth:
                x = layer(x)
            else:
                x = torch.relu(layer(x))
        if self.output_clip is not None:
            x = torch.clamp(x, min=-self.output_clip, max=self.output_clip)
        return x