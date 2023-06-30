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
                 output_dim=128):
        super(ConvPool, self).__init__()

        self.candidate_list = ["GCNConv", "SAGEConv",
                               "GATConv", "GraphConv",
                               "TAGConv", "ARMAConv",
                               "SGConv", "HyperGraphConv",
                               "ClusterGCNConv"]

        self.gcn_conv = GCNConv(input_dim, output_dim)
        self.sage_conv = SAGEConv(input_dim, output_dim)
        self.gat_conv = GATConv(input_dim, output_dim, heads=8, concat=False)
        self.graph_conv = GraphConv(input_dim, output_dim)
        self.tag_conv = TAGConv(input_dim, output_dim)
        self.arma_conv = ARMAConv(input_dim, output_dim)
        self.sg_conv = SGConv(input_dim, output_dim)
        self.hypergraph_conv = HypergraphConv(input_dim, output_dim)
        self.clustergcn_conv = ClusterGCNConv(input_dim, output_dim)

    def get_conv(self, conv_name):

        if conv_name == "GCNConv":
            graph_conv = self.gcn_conv
        elif conv_name == "SAGEConv":
            graph_conv = self.sage_conv
        elif conv_name == "GATConv":
            graph_conv = self.gat_conv
        elif conv_name == "GraphConv":
            graph_conv = self.graph_conv
        elif conv_name == "TAGConv":
            graph_conv = self.tag_conv
        elif conv_name == "ARMAConv":
            graph_conv = self.arma_conv
        elif conv_name == "SGConv":
            graph_conv = self.sg_conv
        elif conv_name == "HyperGraphConv":
            graph_conv = self.hypergraph_conv
        elif conv_name == "ClusterGCNConv":
            graph_conv = self.clustergcn_conv
        else:
            raise Exception("Sorry current version don't "
                            "Support this default graph convolution", conv_name)

        return graph_conv