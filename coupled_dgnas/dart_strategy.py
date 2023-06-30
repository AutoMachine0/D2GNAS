import torch
from planetoid import GraphData
from coupled_dgnas.coupled_supernet import CoupledSuperNet

class DARTS(object):

    def __init__(self,
                 operation_candidates_list,
                 supernet_config,
                 archi_param_optim_config,
                 operation_weight_optim_config,
                 device):

        self.supernet = CoupledSuperNet(supernet_config=supernet_config,
                                        archi_param_optim_config=archi_param_optim_config,
                                        operation_weight_optim_config=operation_weight_optim_config,
                                        device=device)

        self.operation_candidates_list = operation_candidates_list

        self.supernet.coupled_supernet_construction_with_operation_candidates(self.operation_candidates_list)

        self.loss_function = torch.nn.CrossEntropyLoss()

        self.history_gnn_architecture_list = []

    def search(self, graph, search_epoch, return_top_k):

        print("One-shot DARTS Search Strategy Starting")
        print(64*"=")

        for epoch in range(search_epoch):

            y_pre = self.supernet(graph.x, graph.edge_index)

            train_loss = self.loss_function(y_pre[graph.train_mask],
                                            graph.y[graph.train_mask])

            self.supernet.operation_weight_optimizer.zero_grad()
            self.supernet.architecture_parameter_optimizer.zero_grad()

            train_loss.backward()
            self.supernet.operation_weight_optimizer.step()

            y_pre = self.supernet(graph.x, graph.edge_index)

            val_loss = self.loss_function(y_pre[graph.val_mask],
                                          graph.y[graph.val_mask])

            self.supernet.architecture_parameter_optimizer.zero_grad()
            val_loss.backward()
            self.supernet.architecture_parameter_optimizer.step()

            print(32*"+")
            print("Search Epoch", epoch + 1,
                  "Operation Weight Loss", train_loss.item(),
                  "Architecture Parameter Loss:", val_loss.item())

            best_gnn_architecture = self.best_alpha_gnn_architecture_output(self.supernet.alpha_parameters_list)

            print("Best GNN Architecture:", best_gnn_architecture)

        print(64 * "=")


        if int(return_top_k) <= len(self.history_gnn_architecture_list):
            best_gnn_architecture_candidates = self.history_gnn_architecture_list[-int(return_top_k):]
        else:
            best_gnn_architecture_candidates = self.history_gnn_architecture_list

        print("Sampled Top", return_top_k, "GNN Architecture:")

        for gnn_architecture in best_gnn_architecture_candidates:
            print(gnn_architecture)

        print("One-shot DARTS Search Strategy Completion")

        return best_gnn_architecture_candidates

    def best_alpha_gnn_architecture_output(self, alpha_parameters_list):

        best_gnn_architecture = []

        for alpha_list, candidate_list in zip(alpha_parameters_list, self.operation_candidates_list):
            alpha_list = alpha_list.cpu().detach().numpy().tolist()
            best_index = alpha_list.index(max(alpha_list))
            best_gnn_architecture.append(candidate_list[best_index])

        if best_gnn_architecture not in self.history_gnn_architecture_list:
            self.history_gnn_architecture_list.append(best_gnn_architecture)

        return best_gnn_architecture

if __name__=="__main__":

    data_name = "Computers"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = GraphData(data_name, shuffle=False).data

    supernet_config = {"input_dimension": graph.num_node_features,
                       "hidden_dimension": 128,
                       "output_dimension": graph.num_classes,
                       "node_element_dropout_probability": 0.5,
                       "edge_dropout_probability": 0.3}

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

    mydarts = DARTS(operation_candidates_list=operation_candidates_list,
                    supernet_config=supernet_config,
                    operation_weight_optim_config=operation_weight_optim_config,
                    archi_param_optim_config=archi_param_optim_config,
                    device=device)

    mydarts.search(graph, 100, 2)