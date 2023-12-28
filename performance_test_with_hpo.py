import os
import torch
import numpy as np
from planetoid import GraphData
from hpo.hp_search import HpSearchObj
from torch_geometric.loader import ClusterData, ClusterLoader
from multi_trail_gnas.multi_trail_evaluation import MultiTrailEvaluation

def read_top_gnn(search_strategy,
                 data):

    path = os.path.abspath(__file__)[:-len(os.path.basename(__file__))] + "TopGNN/" + search_strategy + "_" + data + ".txt"
    print("Read the TopGNN Information:", path)
    gnn_list = []
    with open(path, "r") as f:
        for gnn in f.readlines():
            gnn_list.append(eval(gnn))
    return gnn_list

def hpo(gnn_architecture,
        graph,
        device,
        tuning_epoch,
        gnn_train_epoch,
        data,
        search_strategy,
        information):

    print("Search Strategy:" + search_strategy + " Dataset:" + data + " HPO Starting")

    HS = HpSearchObj(gnn_architecture=gnn_architecture,
                     gnn_train_epoch=gnn_train_epoch,
                     graph=graph,
                     device=device,
                     information=information)

    learning_rate, weight_decay, node_element_dropout_probability, edge_dropout_probability, hidden_dimension = HS.hp_tuning(tuning_epoch)

    dir = os.path.abspath(__file__)[:-len(os.path.basename(__file__))] + "TopGNNHPO/"

    if not os.path.exists(dir):
        os.makedirs(dir)

    path = dir + search_strategy + "_" + data + ".txt"

    with open(path, "a+") as f:
        f.write("learning_rate=" + str(learning_rate) + " weight_decay=" + str(weight_decay) +
                " node_element_dropout_probability=" + str(node_element_dropout_probability) +
                " edge_dropout_probability=" + str(edge_dropout_probability) +
                " hidden_dimension=" + str(hidden_dimension)+"\n")

    print("Search Strategy:" + search_strategy + " Dataset:" + data + " HPO Completion")

    return learning_rate, weight_decay, node_element_dropout_probability, edge_dropout_probability, hidden_dimension

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
                information):

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

def best_validation_architecture(search_strategy, data, graph, device, gnn_train_epoch):

    cluster_data = ClusterData(graph, num_parts=1)
    graph_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False)

    gnn_model_config = {"num_node_features": graph.num_node_features,
                        "num_classes": graph.num_classes,
                        "hidden_dimension": 128,
                        "learn_rate": 0.001,
                        "node_element_dropout_probability": 0.5,
                        "edge_dropout_probability": 0.5,
                        "weight_decay": 0.005,
                        "train_epoch": gnn_train_epoch}

    best_avg_val_score = 0
    best_val_gnn_architecture = None

    for gnn in read_top_gnn(search_strategy, data):
        avg_val_acc = []
        for epoch in range(5):
            torch.cuda.empty_cache()
            estimator = MultiTrailEvaluation(gnn_model_config=gnn_model_config,
                                             graph=graph_loader,
                                             device=device)
            score = estimator.get_best_validation_estimation(gnn)
            print("Search Strategy:" + search_strategy + " Dataset:" + data)
            print(str(epoch) + " Validation Epoch Validation Accuracy:", score)
            avg_val_acc.append(score)

        avg_val = np.array(avg_val_acc).mean()
        print("GNN Architecture: " + str(gnn) + " Avg Validation Acc: " + str(avg_val))
        if avg_val > best_avg_val_score:
            best_val_gnn_architecture = gnn
            best_avg_val_score = avg_val

    print("The Best Avg Validation GNN Architecture: " + str(best_val_gnn_architecture) + "The Best Avg Validation ACC:" + str(best_avg_val_score))
    return best_val_gnn_architecture

if __name__ == "__main__":

    data_list = ["Computers", "Photo", "Pubmed"]
    strategy_list = ["D2GNAS", "GraphNAS", "AutoGraph", "AutoGNAS", "DeepGNAS", "DDS", "DARTS"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # default config
    tuning_epoch = 200
    gnn_train_epoch = 30
    gnn_test_train_epoch = 100
    test_epoch = 10
    validation_training_epoch = 100

    for strategy in strategy_list:
        for data in data_list:
            graph = GraphData(data, shuffle=False).data
            top_gnn = best_validation_architecture(strategy, data, graph, device, validation_training_epoch)
            print("the best gnn:", top_gnn)
            information = strategy + "_" + data

            learning_rate, weight_decay, node_element_dropout_probability, edge_dropout_probability, hidden_dimension = hpo(gnn_architecture=top_gnn,
                                                                                                                            graph=graph,
                                                                                                                            device=device,
                                                                                                                            tuning_epoch=tuning_epoch,
                                                                                                                            gnn_train_epoch=gnn_train_epoch,
                                                                                                                            data=data,
                                                                                                                            search_strategy=strategy,
                                                                                                                            information=information)

            test_record(graph=graph,
                        gnn_architecture=top_gnn,
                        learning_rate=learning_rate,
                        weight_decay=weight_decay,
                        node_element_dropout_probability=node_element_dropout_probability,
                        edge_dropout_probability=edge_dropout_probability,
                        gnn_train_epoch=gnn_test_train_epoch,
                        hidden_dimension=hidden_dimension,
                        device=device,
                        test_epoch=test_epoch,
                        data=data,
                        search_strategy=strategy,
                        information=information)

