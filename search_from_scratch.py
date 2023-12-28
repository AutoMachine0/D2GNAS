import os
import copy
import time
import torch
from planetoid import GraphData
from coupled_dgnas.dds_strategy import DDS
from coupled_dgnas.dart_strategy import DARTS
from multi_trail_gnas.graphnas_strategy import GraphNAS
from multi_trail_gnas.autograph_strategy import AutoGraph
from multi_trail_gnas.autognas_strategy import AutoGNAS
from multi_trail_gnas.deepgnas_strategy import DeepGNAS
from mixed_supernet import MixedSuperNet
from supernet_pruning_search import SupernetPruningSearch
from torch_geometric.loader import ClusterData, ClusterLoader
from multi_trail_gnas.multi_trail_evaluation import MultiTrailEvaluation

def graphnas(graph, graph_loader, device):
    
    # default configuration
    gnn_model_config = {"num_node_features": graph.num_node_features,
                        "num_classes": graph.num_classes,
                        "hidden_dimension": 128,
                        "learn_rate": 0.005,
                        "node_element_dropout_probability": 0.6,
                        "edge_dropout_probability": 0.5,
                        "weight_decay": 0.0005,
                        "train_epoch": 200}

    estimator = MultiTrailEvaluation(gnn_model_config=gnn_model_config,
                                     graph=graph_loader,
                                     device=device)

    search_parameter = {"controller_hidden_dimension": 100,
                        "controller_optimizer_learn_rate": 0.00035,
                        "controller_optimizer_weight_decay": 0.0001}

    searcher = GraphNAS(estimator,
                        search_parameter,
                        device)

    # default configuration
    top_gnn, _ = searcher.search(controller_train_epoch=1000,
                                 num_sampled_gnn_for_one_train_epoch=1,
                                 scale_of_sampled_gnn=100,
                                 return_top_k=10)

    return top_gnn

def autograph(graph, graph_loader, device):
    
    # default configuration
    gnn_model_config = {"num_node_features": graph.num_node_features,
                        "num_classes": graph.num_classes,
                        "hidden_dimension": 128,
                        "learn_rate": 0.005,
                        "node_element_dropout_probability": 0.6,
                        "edge_dropout_probability": 0.5,
                        "weight_decay": 0.0005,
                        "train_epoch": 200}

    estimator = MultiTrailEvaluation(gnn_model_config=gnn_model_config,
                                     graph=graph_loader,
                                     device=device)

    searcher = AutoGraph(estimator=estimator)

    # default configuration
    top_gnn, _ = searcher.search(num_population=100,
                                 search_epoch=1000,
                                 return_top_k=10)

    return top_gnn

def autognas(graph, graph_loader, device):

    # default configuration
    gnn_model_config = {"num_node_features": graph.num_node_features,
                        "num_classes": graph.num_classes,
                        "hidden_dimension": 128,
                        "learn_rate": 0.005,
                        "node_element_dropout_probability": 0.6,
                        "edge_dropout_probability": 0.5,
                        "weight_decay": 0.0005,
                        "train_epoch": 200}

    estimator = MultiTrailEvaluation(gnn_model_config=gnn_model_config,
                                     graph=graph_loader,
                                     device=device)

    # default configuration
    search_parameter = {"sharing_population_size": 20,
                        "parent_num": 1,
                        "mutation_num": 1}

    searcher = AutoGNAS(estimator=estimator,
                        search_parameter=search_parameter)
    
    # default configuration
    top_gnn, _ = searcher.search(num_population=100, search_epoch=1000, return_top_k=10)

    return top_gnn

def deepgnas(graph, graph_loader, device):

    gnn_model_config = {"num_node_features": graph.num_node_features,
                        "num_classes": graph.num_classes,
                        "hidden_dimension": 128,
                        "learn_rate": 0.0001,
                        "node_element_dropout_probability": 0.6,
                        "edge_dropout_probability": 0.3,
                        "weight_decay": 0.0001,
                        "train_epoch": 200}

    estimator = MultiTrailEvaluation(gnn_model_config=gnn_model_config,
                                     graph=graph_loader,
                                     device=device)
    # default configuration
    agent_train_epoch = 1000
    search_parameter = {"gamma": 1.0,
                        "epsilon": 1.0,
                        "test_epsilon": 0.2,
                        "agent_train_epoch": agent_train_epoch,
                        "hidden_dim": 128,
                        "state_space_vec_dim": 128,
                        "action_space_vec_dim": 128,
                        "target_network_update_epoch": 10}

    searcher = DeepGNAS(estimator=estimator,
                        search_parameter=search_parameter)
    
    # default configuration
    top_gnn, top_gnn_score = searcher.search(agent_train_epoch=agent_train_epoch,
                                             scale_of_sampled_gnn=100,
                                             return_top_k=10)

    return top_gnn

def darts(graph, operation_candidates_list, device):

    supernet_config = {"input_dimension": graph.num_node_features,
                       "hidden_dimension": 128,
                       "output_dimension": graph.num_classes,
                       "node_element_dropout_probability": 0.6,
                       "edge_dropout_probability": 0.5}

    operation_weight_optim_config = {"operation_weight_learn_rate": 0.001,
                                     "operation_weight_weight_decay": 0.0001}

    archi_param_optim_config = {"archi_param_learn_rate": 0.1,
                                "archi_param_weight_decay": 0.001}

    searcher = DARTS(operation_candidates_list=operation_candidates_list,
                     supernet_config=supernet_config,
                     operation_weight_optim_config=operation_weight_optim_config,
                     archi_param_optim_config=archi_param_optim_config,
                     device=device)
    
    # default configuration
    top_gnn = searcher.search(graph=graph,
                              search_epoch=1100,
                              return_top_k=10)

    return top_gnn

def dds(graph, operation_candidates_list, device):

    dynamic_supernet_config = {"ConvPool": {"remain_size": 3},
                               "NormPool": {"remain_size": 2},
                               "ActPool": {"remain_size": 3}}

    supernet_config = {"input_dimension": graph.num_node_features,
                       "hidden_dimension": 128,
                       "output_dimension": graph.num_classes,
                       "node_element_dropout_probability": 0.6,
                       "edge_dropout_probability": 0.5}

    operation_weight_optim_config = {"operation_weight_learn_rate": 0.007,
                                     "operation_weight_weight_decay": 3e-4}

    archi_param_optim_config = {"archi_param_learn_rate": 0.1,
                                "archi_param_weight_decay": 1e-3}

    searcher = DDS(dynamic_supernet_config=dynamic_supernet_config,
                   operation_candidates_list=operation_candidates_list,
                   supernet_config=supernet_config,
                   operation_weight_optim_config=operation_weight_optim_config,
                   archi_param_optim_config=archi_param_optim_config,
                   device=device)

    # default configuration
    top_gnn = searcher.search(graph=graph,
                              inner_search_epoch=1100,
                              return_top_k=10)

    return top_gnn

def d2gnas(graph, operation_candidates_list, device):

    supernet_dim_config = {"input_dimension": graph.num_node_features,
                           "hidden_dimension": 128,
                           "output_dimension": graph.num_classes,
                           "edge_dropout_probability": 0.5,
                           "node_element_dropout_probability": 0.5}

    operation_weight_optimizer_config = {"operation_weight_learn_rate": 0.001,
                                         "operation_weight_weight_decay": 0.005}

    supernet = MixedSuperNet(supernet_dim_config,
                             operation_weight_optimizer_config,
                             device)

    supernet.mixed_supernet_construction_with_operation_candidates(operation_candidates_list)

    loss_f = torch.nn.CrossEntropyLoss()
    
    # default configuration
    supernet_config = {"warm_up_train_epoch": 130,
                       "single_path_training_sample_size_list": [80, 0]}

    differentiable_search_optimizer_config_dict = {"lr": 0.01,
                                                   "decay": 0.005}
    # default configuration
    differentiable_searcher_config = {"temperature": 0.5,
                                      "differentiable_search_optimizer_config_dict": differentiable_search_optimizer_config_dict,
                                      "differentiable_search_epoch_list": [1100, 100],
                                      "differentiable_search_num_return_top_k_gnn": 10}

    searcher = SupernetPruningSearch(supernet=supernet,
                                     loss_f=loss_f,
                                     graph=graph,
                                     supernet_config=supernet_config,
                                     differentiable_searcher_config=differentiable_searcher_config,
                                     device=device)
    top_gnn = searcher.search()

    return top_gnn

def gnn_record(data_name, search_strategy, top_gnn, time_cost):

    dir = os.path.abspath(__file__)[:-len(os.path.basename(__file__))] + "/TopGNN/"

    if not os.path.exists(dir):
        os.makedirs(dir)

    path = dir + search_strategy + "_" + data_name + ".txt"
    with open(path, "w+") as f:
        for gnn in top_gnn:
            f.write(str(gnn)+"\n")

    path = dir + search_strategy + "_" + data_name + "_Time_Cost" + ".txt"
    with open(path, "w") as f:
        f.write(time_cost)

    print("The Search strategy", search_strategy, "Top GNN For The Dataset", data_name, "Recording Completion ")
    print("The Recording Path", path)

if __name__ == "__main__":

    data_list = ["Computers", "Photo", "Pubmed", "CS", "Physics"]
    
    for data_name in data_list:
        graph = GraphData(data_name, shuffle=False).data
        cluster_data = ClusterData(graph, num_parts=1)
        graph_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # t1 = time.time()
        # top_gnn = graphnas(graph=graph, graph_loader=graph_loader, device=device)
        # t2 = time.time()
        # time_cost = str(t2-t1)
        #
        # gnn_record(data_name=data_name,
        #            search_strategy="GraphNAS",
        #            top_gnn=top_gnn,
        #            time_cost=time_cost)
        #
        # t1 = time.time()
        # top_gnn = autograph(graph=graph, graph_loader=graph_loader, device=device)
        # t2 = time.time()
        # time_cost = str(t2 - t1)
        #
        # gnn_record(data_name=data_name,
        #            search_strategy="AutoGraph",
        #            top_gnn=top_gnn,
        #            time_cost=time_cost)

        # t1 = time.time()
        # top_gnn = autognas(graph=graph, graph_loader=graph_loader, device=device)
        # t2 = time.time()
        # time_cost = str(t2 - t1)
        #
        # gnn_record(data_name=data_name,
        #            search_strategy="AutoGNAS",
        #            top_gnn=top_gnn,
        #            time_cost=time_cost)

        # t1 = time.time()
        # top_gnn = deepgnas(graph=graph, graph_loader=graph_loader, device=device)
        # t2 = time.time()
        # time_cost = str(t2 - t1)
        # 
        # gnn_record(data_name=data_name,
        #            search_strategy="DeepGNAS",
        #            top_gnn=top_gnn,
        #            time_cost=time_cost)

        # operation_candidates_list_input = copy.deepcopy(operation_candidates_list)
        #
        # t1 = time.time()
        # top_gnn = darts(graph=graph, operation_candidates_list=operation_candidates_list_input, device=device)
        # t2 = time.time()
        # time_cost = str(t2 - t1)
        #
        # gnn_record(data_name=data_name,
        #            search_strategy="DARTS",
        #            top_gnn=top_gnn,
        #            time_cost=time_cost)
        #
        # operation_candidates_list_input = copy.deepcopy(operation_candidates_list)
        #
        # t1 = time.time()
        # top_gnn = dds(graph=graph, operation_candidates_list=operation_candidates_list_input, device=device)
        # t2 = time.time()
        # time_cost = str(t2 - t1)
        #
        # gnn_record(data_name=data_name,
        #            search_strategy="DDS",
        #            top_gnn=top_gnn,
        #            time_cost=time_cost)

        operation_candidates_list_input = copy.deepcopy(operation_candidates_list)

        t1 = time.time()
        top_gnn = d2gnas(graph=graph,
                         operation_candidates_list=operation_candidates_list_input,
                         device=device)
        t2 = time.time()
        time_cost = str(t2 - t1)
        print("D2GNAS Search Time Cost:", time_cost)

        gnn_record(data_name=data_name,
                   search_strategy="D2GNAS",
                   top_gnn=top_gnn,
                   time_cost=time_cost)

