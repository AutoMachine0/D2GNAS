import torch
from torch_geometric.nn import GraphNorm, \
                               InstanceNorm, \
                               LayerNorm, \
                               BatchNorm

class NormPool(torch.nn.Module):

    def __init__(self,
                 input_dim=128):

        super(NormPool, self).__init__()

        self.candidate_list = ["GraphNorm", "InstanceNorm",
                               "LayerNorm", "BatchNorm",
                               "LinearNorm"]

        self.graph_norm = GraphNorm(input_dim)
        self.instance_norm = InstanceNorm(input_dim)
        self.layer_norm = LayerNorm(input_dim)
        self.batch_norm = BatchNorm(input_dim)
        self.linear_norm = LinearNorm()

    def get_candidate(self, candidate_name):

        if candidate_name == "GraphNorm":
            norm_operation = self.graph_norm
        elif candidate_name == "InstanceNorm":
            norm_operation = self.instance_norm
        elif candidate_name == "LayerNorm":
            norm_operation = self.layer_norm
        elif candidate_name == "BatchNorm":
            norm_operation = self.batch_norm
        elif candidate_name == "LinearNorm":
            norm_operation = self.linear_norm
        else:
            raise Exception("Sorry current version don't "
                            "Support this default graph norm", candidate_name)
        return norm_operation

    def forward(self, x):

        return self.norm_operation(x)

class LinearNorm(torch.nn.Module):
    def __init__(self):
        super(LinearNorm, self).__init__()

    def forward(self, x):

        return x