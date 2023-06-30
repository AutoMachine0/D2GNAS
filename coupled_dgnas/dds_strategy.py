import copy
import torch
import numpy as np
from planetoid import GraphData
from coupled_dgnas.coupled_supernet import CoupledSuperNet

class DDS(object):

    def __init__(self,
                 dynamic_supernet_config,
                 operation_candidates_list,
                 supernet_config,
                 archi_param_optim_config,
                 operation_weight_optim_config,
                 device):

        self.dynamic_supernet_config = dynamic_supernet_config

        self.supernet_config = supernet_config
        self.archi_param_optim_config = archi_param_optim_config
        self.operation_weight_optim_config = operation_weight_optim_config
        self.device = device
        self.operation_candidates_list = operation_candidates_list

        self.supernet = CoupledSuperNet(supernet_config=self.supernet_config,
                                        archi_param_optim_config=self.archi_param_optim_config,
                                        operation_weight_optim_config=self.operation_weight_optim_config,
                                        device=device)

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.outer_loop_history_gnn_architecture_list = []

    def search(self, graph, inner_search_epoch, return_top_k):

        print("One-shot DDS Search Strategy Starting")
        print(64*"=")

        self.dynamic_supernet = []
        self.outer_epoch = 1
        while len(self.operation_candidates_list[0]) != 0:

            print(self.outer_epoch, "Outer Loop Starting New Dynamic Hyernetwork Construction")
            # get dynamic supernet operation candidates list
            self.dynamic_supernet = self.dynamic_supernet_construction(self.dynamic_supernet)
            # based on dynamic supernet operation candidates list construct supernet
            self.supernet.coupled_supernet_construction_with_operation_candidates(self.dynamic_supernet)
            # delete the inner loop best gnn architecture list
            self.inner_loop_history_gnn_architecture_list = []
            print("Dynamic supernet Construction Completion:", self.dynamic_supernet)
            print(self.outer_epoch, "Outer Loop Completion")

            print(self.outer_epoch, "Inner Loop Starting")
            best_gnn_architecture = None

            for epoch in range(inner_search_epoch):
                # operation weights update based on train gradient
                y_pre = self.supernet(graph.x, graph.edge_index)
                train_loss = self.loss_function(y_pre[graph.train_mask],
                                                graph.y[graph.train_mask])
                self.supernet.operation_weight_optimizer.zero_grad()
                self.supernet.architecture_parameter_optimizer.zero_grad()
                train_loss.backward()
                self.supernet.operation_weight_optimizer.step()

                # architecture parameters update validation gradient
                y_pre = self.supernet(graph.x, graph.edge_index)
                val_loss = self.loss_function(y_pre[graph.val_mask],
                                              graph.y[graph.val_mask])
                self.supernet.architecture_parameter_optimizer.zero_grad()
                val_loss.backward()
                self.supernet.architecture_parameter_optimizer.step()

                print(32*"+")
                print("Inner Loop Search Epoch", epoch + 1,
                      "Operation Weight Loss", train_loss.item(),
                      "Architecture Parameter Loss:", val_loss.item())

                best_gnn_architecture = self.best_alpha_gnn_architecture_output(self.supernet.alpha_parameters_list)
                print("Best GNN Architecture In This Inner Loop:", best_gnn_architecture)
                print(32 * "+")

            # retain the optimal candidates in the dynamic supernet before enter the outer loop
            self.dynamic_supernet = []
            for operation in best_gnn_architecture:
                self.dynamic_supernet.append([operation])
            # collect the optimal gnn architecture in this inner loop
            self.outer_loop_history_gnn_architecture_list.append(copy.deepcopy(self.inner_loop_history_gnn_architecture_list))

            print(self.outer_epoch, "Inner Loop Completion")
            self.outer_epoch += 1

        print(64 * "=")

        group = self.return_top_k_divide(return_top_k)
        print("Sub Return Top K Group From Inner Loop History GNN:", group)
        # reverse outer loop history gnn, make later inner loop history gnn near the front.
        self.outer_loop_history_gnn_architecture_list.reverse()
        best_gnn_architecture_candidates_list = []

        for sub_return_top_k, best_gnn_architecture_list in zip(group, self.outer_loop_history_gnn_architecture_list):

            if sub_return_top_k <= len(best_gnn_architecture_list):
                best_gnn_architecture_candidates = best_gnn_architecture_list[-sub_return_top_k:]
            else:
                best_gnn_architecture_candidates = best_gnn_architecture_list

            best_gnn_architecture_candidates_list = best_gnn_architecture_candidates_list + \
                                                    best_gnn_architecture_candidates

        print("Sampled Top", return_top_k, "GNN Architecture:")

        for gnn_architecture in best_gnn_architecture_candidates_list:
            print(gnn_architecture)

        print("One-shot DDS Search Strategy Completion")

        return best_gnn_architecture_candidates_list

    def dynamic_supernet_construction(self, dynamic_supernet):

        num_dynamic_supernet_config = len(self.dynamic_supernet_config)

        # dynamic supernet candidates initialization
        if dynamic_supernet == []:
            # based on operation_candidates_list get supernet　component object and corresponding operation candidates
            for index_operation, operation_candidates in zip(range(len(self.operation_candidates_list)),
                                                                       self.operation_candidates_list):
                # based on num_dynamic_supernet_config get dynamic supernet configuration
                # each component includes some operation candidates
                for index, dynamic_config_name in zip(range(num_dynamic_supernet_config), self.dynamic_supernet_config):
                    # based on index_operation and corresponding index control the right dynamic_config to tackle right operation_candidates
                    # for getting right dynamic_component_candidates to construct dynamic_supernet
                    if (index_operation == index) or ((index_operation - num_dynamic_supernet_config) == index):

                        dynamic_component_candidates = []
                        # random select operation_candidates in the operation candidates to construct dynamic_supernet
                        index_array = np.array([i for i in range(len(operation_candidates))])
                        dynamic_config = self.dynamic_supernet_config[dynamic_config_name]
                        operation_size = dynamic_config["remain_size"]
                        index_list = np.random.choice(index_array, size=operation_size, replace=False).tolist()

                        for index in index_list:
                            dynamic_component_candidates.append(operation_candidates[index])

                        # dynamic supernet initialization
                        dynamic_supernet.append(dynamic_component_candidates)

                        self.operation_candidates_list[index_operation] = np.delete(operation_candidates, index_list)
        else:

            for index_operation, operation_candidates, remain_operation_candidates in zip(range(len(self.operation_candidates_list)),
                                                                                                    self.operation_candidates_list,
                                                                                                    dynamic_supernet):

                for index, dynamic_config_name in zip(range(num_dynamic_supernet_config), self.dynamic_supernet_config):

                    if (index_operation == index) or ((index_operation - num_dynamic_supernet_config) == index):

                        index_array = np.array([i for i in range(len(operation_candidates))])
                        dynamic_config = self.dynamic_supernet_config[dynamic_config_name]
                        operation_size = dynamic_config["remain_size"]

                        index_list = []
                        temp_index = None

                        while len(remain_operation_candidates) < operation_size:
                            index = np.random.choice(index_array, size=1, replace=False).tolist()[0]
                            while index == temp_index:
                                index = np.random.choice(index_array, size=1, replace=False).tolist()[0]
                            remain_operation_candidates.append(operation_candidates[index])
                            index_list.append(index)
                            if len(operation_candidates) == 1:
                                break
                            temp_index = index

                        self.operation_candidates_list[index_operation] = np.delete(operation_candidates, index_list)

        return dynamic_supernet

    def best_alpha_gnn_architecture_output(self, alpha_parameters_list):

        best_gnn_architecture = []
        for alpha_list, candidate_list in zip(alpha_parameters_list, self.dynamic_supernet):
            alpha_list = alpha_list.cpu().detach().numpy().tolist()
            best_index = alpha_list.index(max(alpha_list))
            best_gnn_architecture.append(candidate_list[best_index])

        if best_gnn_architecture not in self.inner_loop_history_gnn_architecture_list:
            self.inner_loop_history_gnn_architecture_list.append(best_gnn_architecture)

        return best_gnn_architecture

    def return_top_k_divide(self, return_top_k):

        if return_top_k < (self.outer_epoch-1):
            return_top_k = self.outer_epoch - 1

        group = []

        while return_top_k != 0:
            if group == []:
                for _ in range(self.outer_epoch-1):
                    group.append(1)
                return_top_k = return_top_k - self.outer_epoch + 1
            else:
                for index in range(len(group)):
                    if return_top_k != 0:
                        group[index] = group[index] + 1
                        return_top_k = return_top_k - 1
                    else:
                        break
        return group

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
                                 ["GraphNorm", "InstanceNorm",
                                  "LayerNorm", "BatchNorm",
                                  "LinearNorm"],
                                 ["Elu", "LeakyRelu",
                                  "Relu", "Relu6",
                                  "Sigmoid", "Softplus",
                                  "Tanh", "Linear"]]

    dynamic_supernet_config = {"ConvPool": {"remain_size": 3},
                                   "NormPool": {"remain_size": 2},
                                   "ActPool": {"remain_size": 3}}

    archi_param_optim_config = {"archi_param_learn_rate": 0.1,
                                "archi_param_weight_decay": 0.01}

    operation_weight_optim_config = {"operation_weight_learn_rate": 0.001,
                                     "operation_weight_weight_decay": 0.0001}

    mydds = DDS(dynamic_supernet_config=dynamic_supernet_config,
                operation_candidates_list=operation_candidates_list,
                supernet_config=supernet_config,
                operation_weight_optim_config=operation_weight_optim_config,
                archi_param_optim_config=archi_param_optim_config,
                device=device)

    mydds.search(graph, 10, 5)