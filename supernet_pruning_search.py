import os
import torch
import numpy as np
from mixed_supernet import MixedSuperNet
from planetoid import GraphData
from search_strategy.differentiable import DifferentiableSearch

class SupernetPruningSearch(object):

    def __init__(self,
                 supernet,
                 loss_f,
                 graph,
                 supernet_config,
                 differentiable_searcher_config,
                 device):

        self.supernet = supernet
        self.loss_f = loss_f
        self.graph = graph

        self.warm_up_training_epoch = supernet_config["warm_up_train_epoch"]
        self.single_path_training_sample_size_list = supernet_config["single_path_training_sample_size_list"]

        self.temperature = differentiable_searcher_config["temperature"]
        self.differentiable_search_optimizer_config_dict = differentiable_searcher_config["differentiable_search_optimizer_config_dict"]
        self.differentiable_search_epoch_list = differentiable_searcher_config["differentiable_search_epoch_list"]
        self.differentiable_search_num_return_top_k_gnn = differentiable_searcher_config["differentiable_search_num_return_top_k_gnn"]

        self.differentiable_searcher = DifferentiableSearch(supernet=self.supernet,
                                                            graph=self.graph,
                                                            differentiable_search_optimizer_config_dict=self.differentiable_search_optimizer_config_dict,
                                                            temperature=self.temperature,
                                                            device=device)

    def warm_up_training(self):

        self.supernet.train()
        for epoch in range(self.warm_up_training_epoch):
            y_pre = self.supernet.mixed_forward(self.graph.x, self.graph.edge_index)

            loss = self.loss_f(y_pre[self.graph.train_mask],
                               self.graph.y[self.graph.train_mask])

            self.supernet.operation_weight_optimizer.zero_grad()
            loss.backward()
            self.supernet.operation_weight_optimizer.step()

            print("Supernet Warm Up Training Epoch", epoch + 1,
                  "Supernet Weights Total Loss:", loss.item())

    def uniform_random_single_path_sample(self, search_paths, sample_size):

        space_size = len(search_paths)
        # sample mode is replace=True
        uniform_random_sample_index = np.random.choice(range(space_size), size=sample_size, replace=True)

        uniform_random_sample_gnn_architecture_list = []

        for index in uniform_random_sample_index:
            uniform_random_sample_gnn_architecture_list.append(search_paths[index])

        return uniform_random_sample_gnn_architecture_list

    def single_path_training(self, search_paths, sample_size):

        uniform_random_sample_gnn_architecture_list = self.uniform_random_single_path_sample(search_paths,
                                                                                             sample_size)
        for sample_gnn_architecture in uniform_random_sample_gnn_architecture_list:
            self.supernet.single_path_architecture_construction(sample_gnn_architecture)
            y_pre = self.supernet.single_path_forward(self.graph.x, self.graph.edge_index)
            train_loss = self.loss_f(y_pre[self.graph.train_mask],
                                     self.graph.y[self.graph.train_mask])

            self.supernet.operation_weight_optimizer.zero_grad()
            train_loss.backward()
            self.supernet.operation_weight_optimizer.step()

    def whole_search_paths_read(self):

        dir = os.getcwd() + "/search_space_gnn_candidates" + ".txt"

        with open(dir, "r") as f:
            whole_search_paths = f.readlines()
            whole_search_paths = [path.replace("\n", "").split(" ") for path in whole_search_paths]
        return whole_search_paths

    def search(self):

        self.warm_up_training()

        search_paths = self.whole_search_paths_read()
        top_gnn_list = []
        for differentiable_search_epoch, single_path_training_sample_size in zip(self.differentiable_search_epoch_list,
                                                                                 self.single_path_training_sample_size_list):

            top_gnn_list, search_paths = self.differentiable_searcher.search(supernet=self.supernet,
                                                                             search_paths=search_paths,
                                                                             search_epoch=differentiable_search_epoch,
                                                                             return_top_k=self.differentiable_search_num_return_top_k_gnn)

            self.single_path_training(search_paths=search_paths, sample_size=single_path_training_sample_size)

        return top_gnn_list

if __name__=="__main__":

    data_name = "CS"
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
                                 ["GraphNorm", "InstanceNorm",
                                  "LayerNorm", "BatchNorm",
                                  "LinearNorm"],
                                 ["Elu", "LeakyRelu",
                                  "Relu", "Relu6",
                                  "Sigmoid", "Softplus",
                                  "Tanh", "Linear"]]

    operation_weight_optimizer_config = {"operation_weight_learn_rate": 0.01,
                                         "operation_weight_weight_decay": 0.0001}

    supernet = MixedSuperNet(supernet_dim_config,
                             operation_weight_optimizer_config,
                             device)

    supernet.mixed_supernet_construction_with_operation_candidates(operation_candidates_list)

    loss_f = torch.nn.CrossEntropyLoss()

    supernet_config = {"warm_up_train_epoch": 10,
                       "single_path_training_sample_size_list": [100, 0]}

    differentiable_search_optimizer_config_dict = {"lr": 0.1,
                                                   "decay": 0.005}

    differentiable_searcher_config = {"temperature": 0.1,
                                      "differentiable_search_optimizer_config_dict": differentiable_search_optimizer_config_dict,
                                      "differentiable_search_epoch_list": [500, 200],
                                      "differentiable_search_num_return_top_k_gnn": 5}

    searcher = SupernetPruningSearch(supernet=supernet,
                                     loss_f=loss_f,
                                     graph=graph,
                                     supernet_config=supernet_config,
                                     differentiable_searcher_config=differentiable_searcher_config,
                                     device=device)
    searcher.search()