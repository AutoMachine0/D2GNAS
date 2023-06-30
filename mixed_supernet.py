import torch
import torch.nn.functional as F
from search_space.mlp import MLP
from graph_augment.edge_dropout_pyg import dropout_edge
from mixed_search_space_with_forward.act_pool import ActPool
from mixed_search_space_with_forward.norm_pool import NormPool
from mixed_search_space_with_forward.conv_pool import ConvPool

from planetoid import GraphData

class MixedSuperNet(torch.nn.Module):

    def __init__(self,
                 supernet_config,
                 operation_weight_optim_config,
                 device):

        super(MixedSuperNet, self).__init__()

        self.input_dimension = supernet_config["input_dimension"]
        self.hidden_dimension = supernet_config["hidden_dimension"]
        self.output_dimension = supernet_config["output_dimension"]
        self.edge_probability = supernet_config["edge_dropout_probability"]
        self.node_element_probability = supernet_config["node_element_dropout_probability"]

        self.operation_weight_learn_rate = operation_weight_optim_config["operation_weight_learn_rate"]
        self.operation_weight_weight_decay = operation_weight_optim_config["operation_weight_weight_decay"]
        self.device = device

        # pre process mlp initialization
        self.pre_process_mlp = MLP(input_dim=self.input_dimension,
                                   output_dim=self.hidden_dimension).to(device)
        # post process mlp initialization
        self.post_process_mlp = MLP(input_dim=self.hidden_dimension,
                                    output_dim=self.output_dimension).to(device)

        # supernet operation pool construction initialization
        # convolution pool initialization
        layer1_conv_pool = ConvPool(self.hidden_dimension, self.hidden_dimension).to(device)
        layer2_conv_pool = ConvPool(self.hidden_dimension, self.hidden_dimension).to(device)

        # normalization pool initialization
        layer1_norm_pool = NormPool(self.hidden_dimension).to(device)
        layer2_norm_pool = NormPool(self.hidden_dimension).to(device)

        # activation pool initialization
        layer1_act_pool = ActPool().to(device)
        layer2_act_pool = ActPool().to(device)

        self.supernet_operation_pool = [layer1_conv_pool, layer1_norm_pool, layer1_act_pool,
                                            layer2_conv_pool, layer2_norm_pool, layer2_act_pool]

        # coupled supernet candidate
        self.conv_candidate = ConvPool().candidate_list
        self.norm_candidate = NormPool().candidate_list
        self.act_candidate = ActPool().candidate_list

        self.num_gnn_layer = 2
        self.component_candidate_dict = {"Convolution": self.conv_candidate,
                                         "Normalization": self.norm_candidate,
                                         "Activation": self.act_candidate}

    def mixed_supernet_construction_with_operation_candidates(self, operation_candidates_list):

        self.mixed_supernet = []

        operation_weights = []

        for operation_candidate, operation_pool in zip(operation_candidates_list,
                                                       self.supernet_operation_pool):

            mix_operation = []

            for operation in operation_candidate:
                operation_obj = operation_pool.get_candidate(operation)
                mix_operation.append(operation_obj)

                if type(operation_pool).__name__ != "ActPool":
                    operation_weights.append({"params": operation_obj.parameters()})

            self.mixed_supernet.append(mix_operation)

        self.operation_weight_optimizer = torch.optim.Adam(operation_weights,
                                                           lr=self.operation_weight_learn_rate,
                                                           weight_decay=self.operation_weight_weight_decay)

    def mixed_forward(self, x, edge_index):

        x = self.pre_process_mlp(x)

        for mix_operation in self.mixed_supernet:
            operation_output_list = []

            for operation in mix_operation:
                try:
                    if "Conv" in type(operation).__name__:
                        drop_edge_index = dropout_edge(edge_index=edge_index, p=self.edge_probability)[0]
                        drop_x = F.dropout(x, p=self.node_element_probability)
                        operation_output_list.append(operation(drop_x, drop_edge_index))
                        continue
                    elif "Norm" in type(operation).__name__:
                        operation_output_list.append(operation(x))
                        continue
                    else:
                        operation_output_list.append(operation(x))
                        continue
                except:
                    raise Exception("Sorry current version don't "
                                    "Support this default operation", type(operation).__name__)

            # calculate mix operation output
            x = sum(operation_output_list)

        x = self.post_process_mlp(x)

        return x

    def single_path_architecture_construction(self, architecture):

        print(architecture, "single path training")

        self.gnn_architecture = architecture

        self.layer1_conv = self.supernet_operation_pool[0].get_candidate(architecture[0])
        self.layer1_norm = self.supernet_operation_pool[1].get_candidate(architecture[1])
        self.layer1_act = self.supernet_operation_pool[2].get_candidate(architecture[2])

        self.layer2_conv = self.supernet_operation_pool[3].get_candidate(architecture[3])
        self.layer2_norm = self.supernet_operation_pool[4].get_candidate(architecture[4])
        self.layer2_act = self.supernet_operation_pool[5].get_candidate(architecture[5])

    def single_path_forward(self, x, edge_index):
        x = self.pre_process_mlp(x)

        drop_edge_index = dropout_edge(edge_index=edge_index, p=self.edge_probability)[0]
        drop_x = F.dropout(x, p=self.node_element_probability)

        x = self.layer1_conv(drop_x, drop_edge_index)
        x = self.layer1_norm(x)
        x = self.layer1_act(x)

        drop_edge_index = dropout_edge(edge_index=edge_index, p=self.edge_probability)[0]
        drop_x = F.dropout(x, p=self.node_element_probability)

        x = self.layer2_conv(drop_x, drop_edge_index)
        x = self.layer2_norm(x)
        x = self.layer2_act(x)

        x = self.post_process_mlp(x)

        return x

if __name__=="__main__":

    data_name = "Computers"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = GraphData(data_name, shuffle=False).data

    supernet_dim_config = {"input_dimension": graph.num_node_features,
                           "hidden_dimension": 128,
                           "output_dimension": graph.num_classes,
                           "edge_dropout_probability": 0.3,
                           "node_element_dropout_probability": 0.5}

    operation_candidates_list = [["GCNConv", "SAGEConv",
                                  "GATConv", "GraphConv",
                                  "TAGConv", "ARMAConv",
                                  "SGConv", "HyperGraphConv",
                                  "ClusterGCNConv"],
                                 ["GraphNorm", "InstanceNorm",
                                  "LayerNorm", "BatchNorm",
                                  "LinearNorm"],
                                 ["Elu", "LeakyRelu",
                                  "Relu", "Relu6",
                                  "Sigmoid", "Softplus",
                                  "Tanh", "Linear"],
                                 ["GCNConv", "SAGEConv",
                                  "GATConv", "GraphConv",
                                  "TAGConv", "ARMAConv",
                                  "SGConv", "HyperGraphConv",
                                  "ClusterGCNConv"],
                                 ["GraphNorm", "InstanceNorm"],
                                 ["Elu", "LeakyRelu",
                                  "Relu", "Relu6",
                                  "Sigmoid", "Softplus",
                                  "Tanh", "Linear"]]


    operation_weight_optim_config = {"operation_weight_learn_rate": 0.01,
                                     "operation_weight_weight_decay": 0.0001}

    my_supernet = MixedSuperNet(supernet_dim_config,
                                operation_weight_optim_config,
                                device)

    my_supernet.mixed_supernet_construction_with_operation_candidates(operation_candidates_list)
    loss_f = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        y_pre = my_supernet.mixed_forward(graph.x, graph.edge_index)

        train_loss = loss_f(y_pre[graph.train_mask],
                            graph.y[graph.train_mask])

        my_supernet.operation_weight_optimizer.zero_grad()
        train_loss.backward()
        my_supernet.operation_weight_optimizer.step()

        print("Train Epoch", epoch+1, "Hypernetwork Weight Loss:", train_loss.item())

    print("Single Path Training")

    for epoch in range(100):

        architecture = ["GCNConv", "GraphNorm", "Relu",
                        "GATConv", "LayerNorm", "Sigmoid"]
        my_supernet.single_path_architecture_construction(architecture)
        y_pre = my_supernet.single_path_forward(graph.x, graph.edge_index)

        train_loss = loss_f(y_pre[graph.train_mask],
                            graph.y[graph.train_mask])

        my_supernet.operation_weight_optimizer.zero_grad()
        train_loss.backward()
        my_supernet.operation_weight_optimizer.step()
        print("Train Epoch", epoch+1, "Architecture Weight Loss:", train_loss.item())