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
        self.gat_conv = GATConv(input_dim, output_dim)
        self.graph_conv = GraphConv(input_dim, output_dim)
        self.tag_conv = TAGConv(input_dim, output_dim)
        self.arma_conv = ARMAConv(input_dim, output_dim)
        self.sg_conv = SGConv(input_dim, output_dim)
        self.hypergraph_conv = HypergraphConv(input_dim, output_dim)
        self.clustergcn_conv = ClusterGCNConv(input_dim, output_dim)

    def get_candidate(self, candidate_name):

        if candidate_name == "GCNConv":
            conv_operation = self.gcn_conv
        elif candidate_name == "SAGEConv":
            conv_operation = self.sage_conv
        elif candidate_name == "GATConv":
            conv_operation = self.gat_conv
        elif candidate_name == "GraphConv":
            conv_operation = self.graph_conv
        elif candidate_name == "TAGConv":
            conv_operation = self.tag_conv
        elif candidate_name == "ARMAConv":
            conv_operation = self.arma_conv
        elif candidate_name == "SGConv":
            conv_operation = self.sg_conv
        elif candidate_name == "HyperGraphConv":
            conv_operation = self.hypergraph_conv
        elif candidate_name == "ClusterGCNConv":
            conv_operation = self.clustergcn_conv
        else:
            raise Exception("Sorry current version don't "
                            "Support this default graph convolution", candidate_name)

        return conv_operation

    def forward(self, x, edge_index):

        return self.conv_operation(x, edge_index)

if __name__=="__main__":
    a = ConvPool()
    print(type(a).__name__)