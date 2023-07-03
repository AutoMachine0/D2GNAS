import os
import torch
import numpy as np
from planetoid import GraphData
from torch_geometric.loader import ClusterData, ClusterLoader
from multi_trail_gnas.multi_trail_evaluation import MultiTrailEvaluation
from multi_trail_gnas.multi_trail_evaluation_mannul_gnn import MultiTrailEvaluation as MultiTrailEvaluation_manual
from scalable_gnn.pasca_v3 import PaScaV3, edge_index_to_sparse_adj
from scalable_gnn.base_op import x_row_normalization

def test_record(graph,
                gnn_architecture,
                learning_rate,
                weight_decay,
                node_element_dropout_probability,
                edge_dropout_probability,
                gnn_train_epoch,
                hidden_dimension,
                device,
                test_epoch,
                data,
                search_strategy,
                information,
                manner):

    gnn_model_config = {"num_node_features": graph.num_node_features,
                        "num_classes": graph.num_classes,
                        "hidden_dimension": hidden_dimension,
                        "learn_rate": learning_rate,
                        "node_element_dropout_probability": node_element_dropout_probability,
                        "edge_dropout_probability": edge_dropout_probability,
                        "weight_decay": weight_decay,
                        "train_epoch": gnn_train_epoch}
    avg_test_acc = []
    std_test_acc = []

    cluster_data = ClusterData(graph, num_parts=1)
    graph_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False)

    for epoch in range(test_epoch):
        torch.cuda.empty_cache()
        print(information)
        if manner=="manual":
            estimator = MultiTrailEvaluation_manual(gnn_model_config=gnn_model_config,
                                                     graph=graph_loader,
                                                     device=device)
        else:
            estimator = MultiTrailEvaluation(gnn_model_config=gnn_model_config,
                                             graph=graph_loader,
                                             device=device)

        score = estimator.get_test_score(gnn_architecture)
        print("Search Strategy:" + search_strategy + " Dataset:" + data)
        print(str(epoch) + " Test Epoch Test Accuracy:", score)
        avg_test_acc.append(score)
        std_test_acc.append(score)

    avg_test = np.array(avg_test_acc).mean()
    std_test = np.array(avg_test_acc).std()

    dir = os.path.abspath(__file__)[:-len(os.path.basename(__file__))] + "Performance/"

    if not os.path.exists(dir):
        os.makedirs(dir)

    path = dir + search_strategy + "_" + data + ".txt"

    with open(path, "a+") as f:
        f.write("avg test accuracy: " + str(avg_test) + " std test accuracy:" + str(std_test) + "\n")

if __name__ == "__main__":

    # Performance test for GCN,GAT,SAGE,SGC
    # # ++++++++++++++++++++++++++++++++++++
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # gnn_test_train_epoch = 100
    # test_epoch = 10
    # data_list = ["Computers", "Photo", "Pubmed"]
    # # For GCN,SGAE,SGC
    # search_strategy_list = ["GCN", "SAGE", "SGC"]
    # gnn_architecture_list = [["GCNConv", "GraphNorm", "Relu", "GCNConv", "GraphNorm", "Relu"],
    #                          ["SAGEConv", "GraphNorm", "Relu", "SAGEConv", "GraphNorm", "Relu"],
    #                          ["SGConv", "GraphNorm", "Relu", "SGConv", "GraphNorm", "Relu"]]
    # manner = "manual"
    # learning_rate = 0.05
    # weight_decay = 0.0005
    # hidden_dimension = 128
    # node_element_dropout_probability = 0.5
    # edge_dropout_probability = 0.0

    # For GAT
    # search_strategy_list = ["GAT"]
    # gnn_architecture_list = [["GATConv", "GraphNorm", "Elu", "GATConv", "GraphNorm", "Elu"]]
    # manner = "manual"
    # learning_rate = 0.05
    # weight_decay = 0.0005
    # hidden_dimension = 128
    # node_element_dropout_probability = 0.0
    # edge_dropout_probability = 0.0
    # for data in data_list:
    #     for search_strategy, gnn_architecture in zip(search_strategy_list, gnn_architecture_list):
    #         graph = GraphData(data, shuffle=False).data
    #         information = search_strategy + "_" + data
    #         test_record(graph=graph,
    #                     gnn_architecture=gnn_architecture,
    #                     learning_rate=learning_rate,
    #                     weight_decay=weight_decay,
    #                     node_element_dropout_probability=node_element_dropout_probability,
    #                     edge_dropout_probability=edge_dropout_probability,
    #                     gnn_train_epoch=gnn_test_train_epoch,
    #                     hidden_dimension=hidden_dimension,
    #                     device=device,
    #                     test_epoch=test_epoch,
    #                     data=data,
    #                     search_strategy=search_strategy,
    #                     information=information,
    #                     manner=manner)
    # # ++++++++++++++++++++++++++++++++++++

    # Performance test for PaScaV3
    # # ++++++++++++++++++++++++++++++++++++
    # test_iter = 10
    # train_epoch = 100
    # dropout_rate = 0.0
    # hidden_dim = 128
    # learning_rate = 0.05
    # weight_decay = 0.0005
    # norm_name = "GraphNorm"
    # data_list = ["Computers", "Photo", "Pubmed"]
    # for data in data_list:
    #     test_acc_list = []
    #     for iter in range(test_iter):
    #         graph = GraphData(data, shuffle=False).data
    #         x = graph.x
    #         y = graph.y
    #         train_mask = graph.train_mask
    #         test_mask = graph.test_mask
    #         feature_dim = graph.num_node_features
    #         num_classes = graph.num_classes
    #
    #         A = edge_index_to_sparse_adj(x, graph.edge_index)
    #         A = A.numpy()
    #         x = x_row_normalization(x)
    #         x = x.to('cpu').numpy()
    #
    #         pasca3 = PaScaV3(feature_dim,
    #                          num_classes,
    #                          dropout_rate=dropout_rate,
    #                          hidden_dim=hidden_dim,
    #                          learning_rate=learning_rate,
    #                          weight_decay=weight_decay,
    #                          norm_name=norm_name)
    #
    #         test_acc = pasca3.train(x=x,
    #                                 A=A,
    #                                 y=y,
    #                                 train_epoch=train_epoch,
    #                                 train_mask=train_mask,
    #                                 test_mask=test_mask)
    #         test_acc_list.append(test_acc)
    #         print(str(iter + 1) + " Test Epoch: " + str(test_acc))
    #
    #     avg_test = np.array(test_acc_list).mean()
    #     std_test = np.array(test_acc_list).std()
    #
    #     dir = os.path.abspath(__file__)[:-len(os.path.basename(__file__))] + "Performance/"
    #
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
    #
    #     path = dir + "PaScaV3" + "_" + data + ".txt"
    #
    #     with open(path, "a+") as f:
    #         f.write("avg test accuracy: " + str(avg_test) + " std test accuracy:" + str(std_test) + "\n")
    # # ++++++++++++++++++++++++++++++++++++

    # Performance test for the optimal GNN architecture designed by D2GNAS
    # # ++++++++++++++++++++++++++++++++++++
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_list = ["Computers", "Photo", "Pubmed"]
    gnn_test_train_epoch = 100
    search_strategy = "DDGNAS"
    test_epoch = 10
    manner = "gnas"

    gnn_architecture_list = [["ClusterGCNConv", "GraphNorm", "Relu6", "SGConv", "GraphNorm", "Relu6"],
                             ["SAGEConv", "LinearNorm", "Tanh", "ARMAConv", "GraphNorm", "Tanh"],
                             ["TAGConv", "InstanceNorm", "Tanh", "HyperGraphConv", "LinearNorm", "Elu"],
                             ["TAGConv", "BatchNorm", "Relu", "TAGConv", "GraphNorm", "LeakyRelu"],
                             ["GATConv", "InstanceNorm", "Relu6", "SGConv", "LayerNorm", "Tanh"]]

    hp_list =[[5e-3, 1e-4, 0.0, 0.2, 256],
              [1e-3, 1e-5, 0.2, 0.3, 1024],
              [1e-3, 0.0, 0.4, 0.0, 64],
              [1e-2, 1e-3, 0.0, 0.5, 256],
              [1e-3, 5e-5, 0.1, 0.3, 1024]]

    for data, gnn_architecture, hp in zip(data_list, gnn_architecture_list, hp_list):
        graph = GraphData(data, shuffle=False).data
        learning_rate = hp[0]
        weight_decay = hp[1]
        node_element_dropout_probability = hp[2]
        edge_dropout_probability = hp[3]
        hidden_dimension = hp[4]
        information = search_strategy + "_" + data

        test_record(graph=graph,
                    gnn_architecture=gnn_architecture,
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    node_element_dropout_probability=node_element_dropout_probability,
                    edge_dropout_probability=edge_dropout_probability,
                    gnn_train_epoch=gnn_test_train_epoch,
                    hidden_dimension=hidden_dimension,
                    device=device,
                    test_epoch=test_epoch,
                    data=data,
                    search_strategy=search_strategy,
                    information=information,
                    manner=manner)
    # # ++++++++++++++++++++++++++++++++++++

