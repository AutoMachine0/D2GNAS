import torch
from torch_geometric.nn import GCNConv, \
                               SAGEConv,\
                               GATConv, \
                               GraphConv,\
                               TAGConv, \
                               ARMAConv,\
                               SGConv,\
                               HypergraphConv, \
                               ClusterGCNConv

class ConvPool(torch.nn.Module):

    def __init__(self,
                 input_dim=128,
                 output_dim=128,
                 conv_name="GCNConv"):

        super(ConvPool, self).__init__()

        self.candidate_list = ["GCNConv", "SAGEConv",
                               "GATConv", "GraphConv",
                               "TAGConv", "ARMAConv",
                               "SGConv", "HyperGraphConv",
                               "ClusterGCNConv"]

        if conv_name == "GCNConv":
            self.conv_operation = GCNConv(input_dim, output_dim)
        elif conv_name == "SAGEConv":
            self.conv_operation = SAGEConv(input_dim, output_dim)
        elif conv_name == "GATConv":
            self.conv_operation = GATConv(input_dim, output_dim)
        elif conv_name == "GraphConv":
            self.conv_operation = GraphConv(input_dim, output_dim)
        elif conv_name == "TAGConv":
            self.conv_operation = TAGConv(input_dim, output_dim)
        elif conv_name == "ARMAConv":
            self.conv_operation = ARMAConv(input_dim, output_dim)
        elif conv_name == "SGConv":
            self.conv_operation = SGConv(input_dim, output_dim)
        elif conv_name == "HyperGraphConv":
            self.conv_operation = HypergraphConv(input_dim, output_dim)
        elif conv_name == "ClusterGCNConv":
            self.conv_operation = ClusterGCNConv(input_dim, output_dim)
        else:
            raise Exception("Sorry current version don't "
                            "Support this default graph convolution", conv_name)

    def forward(self, x, edge_index):

        return self.conv_operation(x, edge_index)

if __name__=="__main__":
    a = ConvPool()
    print(type(a).__name__)