import torch
import torch.nn as nn
import torch.nn.functional as F
from search_space.mlp import MLP
from torch.autograd import Variable
from graph_augment.edge_dropout_pyg import dropout_edge
from coupled_dgnas.search_space_with_forward.act_pool import ActPool
from coupled_dgnas.search_space_with_forward.conv_pool import ConvPool
from coupled_dgnas.search_space_with_forward.norm_pool import NormPool

from planetoid import GraphData

class CoupledSuperNet(torch.nn.Module):

    def __init__(self,
                 supernet_config,
                 archi_param_optim_config,
                 operation_weight_optim_config,
                 device):

        super(CoupledSuperNet, self).__init__()

        self.input_dimension = supernet_config["input_dimension"]
        self.hidden_dimension = supernet_config["hidden_dimension"]
        self.output_dimension = supernet_config["output_dimension"]
        self.edge_dropout_probability = supernet_config["edge_dropout_probability"]
        self.node_element_dropout_probability = supernet_config["node_element_dropout_probability"]

        self.archi_param_learn_rate = archi_param_optim_config["archi_param_learn_rate"]
        self.archi_param_weight_decay = archi_param_optim_config["archi_param_weight_decay"]

        self.operation_weight_learn_rate = operation_weight_optim_config["operation_weight_learn_rate"]
        self.operation_weight_weight_decay = operation_weight_optim_config["operation_weight_weight_decay"]

        self.device = device
        # coupled supernet candidate
        self.conv_candidate = ConvPool().candidate_list
        self.norm_candidate = NormPool().candidate_list
        self.act_candidate = ActPool().candidate_list

    def coupled_supernet_construction_with_operation_candidates(self, operation_candidates_list):

        # pre process mlp初始化
        self.pre_process_mlp = MLP(input_dim=self.input_dimension,
                                   output_dim=self.hidden_dimension).to(self.device)
        # post process mlp初始化
        self.post_process_mlp = MLP(input_dim=self.hidden_dimension,
                                    output_dim=self.output_dimension).to(self.device)

        self.architecture_parameter_construction_with_operation_candidate(operation_candidates_list)

        self.inference_flow_with_mix_operation = []

        operation_weights = []

        for operation_candidate in operation_candidates_list:

            mix_operation = []

            for operation in operation_candidate:

                if operation in self.conv_candidate:
                    operation_obj = ConvPool(conv_name=operation).to(self.device)
                elif operation in self.norm_candidate:
                    operation_obj = NormPool(norm_name=operation).to(self.device)
                elif operation in self.act_candidate:
                    operation_obj = ActPool(act_name=operation).to(self.device)
                else:
                    raise Exception("Sorry current version don't "
                                    "Support this operation", operation)

                operation_weights.append({"params": operation_obj.parameters()})

                mix_operation.append(operation_obj)

            self.inference_flow_with_mix_operation.append(mix_operation)

        self.operation_weight_optimizer = torch.optim.Adam(operation_weights,
                                                           lr=self.operation_weight_learn_rate,
                                                           weight_decay=self.operation_weight_weight_decay)

    def architecture_parameter_construction_with_operation_candidate(self, operation_candidates_list):

        self.alpha_parameters_list = []

        for operation_candidate in operation_candidates_list:
            num_operation_candidates = len(operation_candidate)

            alpha_parameters = Variable(nn.init.uniform_(torch.Tensor(num_operation_candidates))).to(self.device)
            alpha_parameters.requires_grad = True
            nn.init.uniform_(alpha_parameters)
            self.alpha_parameters_list.append(alpha_parameters)

        self.architecture_parameter_optimizer = torch.optim.Adam(self.alpha_parameters_list,
                                                                 lr=self.archi_param_learn_rate,
                                                                 weight_decay=self.archi_param_weight_decay)

    def forward(self, x, edge_index):

        x = self.pre_process_mlp(x)

        for mix_operation, alpha_operation in zip(self.inference_flow_with_mix_operation, self.alpha_parameters_list):

            alpha_operation = F.softmax(alpha_operation, dim=-1)
            operation_output_list = []

            for operation, alpha in zip(mix_operation, alpha_operation):
                if type(operation).__name__ == "ConvPool":
                    drop_edge_index = dropout_edge(edge_index=edge_index, p=self.edge_dropout_probability)[0]
                    drop_x = F.dropout(x, p=self.node_element_dropout_probability)
                    operation_output_list.append(operation(drop_x, drop_edge_index) * alpha)
                    continue
                elif type(operation).__name__ == "NormPool":
                    operation_output_list.append(operation(x) * alpha)
                    continue
                elif type(operation).__name__ == "ActPool":
                    operation_output_list.append(operation(F.dropout(x, p=0.5) * alpha))
                    continue
                else:
                    raise Exception("Sorry current version don't "
                                    "Support this default operation", type(operation).__name__)
            # 计算mix operation输出
            x = sum(operation_output_list)

        x = self.post_process_mlp(x)

        return x

if __name__=="__main__":

    data_name = "Computers"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = GraphData(data_name, shuffle=False).data
    supernet_dim_config = {"input_dimension": graph.num_node_features,
                           "hidden_dimension": 128,
                           "output_dimension": graph.num_classes,
                           "dropout_probability": 0.5}

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

    archi_param_optim_config = {"archi_param_learn_rate": 0.1,
                                "archi_param_weight_decay": 0.01}

    operation_weight_optim_config = {"operation_weight_learn_rate": 0.001,
                                     "operation_weight_weight_decay": 0.0001}

    my_supernet = CoupledSuperNet(supernet_dim_config,
                                  archi_param_optim_config,
                                  operation_weight_optim_config,
                                  device)

    my_supernet.coupled_supernet_construction_with_operation_candidates(operation_candidates_list)
    loss_f = torch.nn.CrossEntropyLoss()

    for epoch in range(50):
        y_pre = my_supernet(graph.x, graph.edge_index)

        train_loss = loss_f(y_pre[graph.train_mask],
                            graph.y[graph.train_mask])

        my_supernet.operation_weight_optimizer.zero_grad()
        my_supernet.architecture_parameter_optimizer.zero_grad()

        train_loss.backward()
        my_supernet.operation_weight_optimizer.step()

        print("Train Epoch", epoch+1, "Supernet Weight Loss:", train_loss.item())

        y_pre = my_supernet(graph.x, graph.edge_index)

        val_loss = loss_f(y_pre[graph.val_mask],
                          graph.y[graph.val_mask])

        my_supernet.architecture_parameter_optimizer.zero_grad()
        val_loss.backward()
        my_supernet.architecture_parameter_optimizer.step()
        print("Train Epoch", epoch+1, "Architecture Parameter Loss:", val_loss.item())