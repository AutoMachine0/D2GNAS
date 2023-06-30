import torch
from torch_geometric.nn import GraphNorm, \
                               InstanceNorm, \
                               LayerNorm, \
                               BatchNorm

class NormPool(torch.nn.Module):

    def __init__(self, input_dim=128):
        super(NormPool, self).__init__()

        self.candidate_list = ["GraphNorm", "InstanceNorm",
                               "LayerNorm", "BatchNorm",
                               "LinearNorm"]

        self.graph_norm = GraphNorm(input_dim)
        self.instance_norm = InstanceNorm(input_dim)
        self.layer_norm = LayerNorm(input_dim)
        self.batch_norm = BatchNorm(input_dim)
        self.linear_norm = LinearNorm()

    def get_norm(self, norm_name):

        if norm_name == "GraphNorm":
            norm = self.graph_norm
        elif norm_name == "InstanceNorm":
            norm = self.instance_norm
        elif norm_name == "LayerNorm":
            norm = self.layer_norm
        elif norm_name == "BatchNorm":
            norm = self.batch_norm
        elif norm_name == "LinearNorm":
            norm = self.linear_norm
        else:
            raise Exception("Sorry current version don't "
                            "Support this default graph norm", norm_name)
        return norm

class LinearNorm(torch.nn.Module):
    def __init__(self):
        super(LinearNorm, self).__init__()

    def forward(self, x):
        return x