import time
import torch
from hyperopt import fmin, tpe
from planetoid import GraphData
from torch_geometric.loader import ClusterData, ClusterLoader
from hpo.hp_search_space import HP_SEARCH_SPACE, HP_SEARCH_SPACE_Mapping
from multi_trail_gnas.multi_trail_evaluation import MultiTrailEvaluation

class HpSearchObj(object):

    def __init__(self,
                 gnn_architecture,
                 gnn_train_epoch,
                 graph,
                 device,
                 information):

        self.gnn_architecture = gnn_architecture
        self.gnn_train_epoch = gnn_train_epoch
        self.graph = graph
        self.device = device
        cluster_data = ClusterData(graph, num_parts=1)
        self.graph_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False)
        self.hpo_epoch = 1
        self.information = information

    def tuning_obj(self, hp_space):

        learning_rate = hp_space["learning_rate"]
        weight_decay = hp_space["weight_decay"]
        node_element_dropout_probability = hp_space["node_element_dropout_probability"]
        edge_dropout_probability = hp_space["edge_dropout_probability"]
        hidden_dimension = hp_space["hidden_dimension"]

        print(16 * "+", str(self.hpo_epoch)+" HPO Epoch, The Hyper Parameter For GNN Training As Follows:")
        print("learning_rate: %f / weight_decay: %f / node_element_dropout_probability: %f / hidden_dimension: %f" % (learning_rate,
                                                                                                        weight_decay,
                                                                                                        node_element_dropout_probability,
                                                                                                        hidden_dimension))
        self.hpo_epoch += 1
        print(self.information)

        gnn_model_config = {"num_node_features": self.graph.num_node_features,
                            "num_classes": self.graph.num_classes,
                            "hidden_dimension": hidden_dimension,
                            "learn_rate": learning_rate,
                            "node_element_dropout_probability": node_element_dropout_probability,
                            "edge_dropout_probability": edge_dropout_probability,
                            "weight_decay": weight_decay,
                            "train_epoch": self.gnn_train_epoch}

        estimator = MultiTrailEvaluation(gnn_model_config=gnn_model_config,
                                         graph=self.graph_loader,
                                         device=self.device)

        score = estimator.get_best_validation_estimation(self.gnn_architecture)

        return -score

    def hp_tuning(self, search_epoch):

        print("target_gnn_architecture:", self.gnn_architecture)

        best_hp = fmin(fn=self.tuning_obj,
                       space=HP_SEARCH_SPACE,
                       algo=tpe.suggest,
                       max_evals=search_epoch)

        learning_rate_index = best_hp["learning_rate"]
        weight_decay_index = best_hp["weight_decay"]
        node_element_dropout_probability_index = best_hp["node_element_dropout_probability"]
        edge_dropout_probability_index = best_hp["edge_dropout_probability"]
        hidden_dimension_index = best_hp["hidden_dimension"]

        learning_rate = HP_SEARCH_SPACE_Mapping["learning_rate"][learning_rate_index]
        weight_decay = HP_SEARCH_SPACE_Mapping["weight_decay"][weight_decay_index]
        node_element_dropout_probability = HP_SEARCH_SPACE_Mapping["node_element_dropout_probability"][node_element_dropout_probability_index]
        edge_dropout_probability = HP_SEARCH_SPACE_Mapping["edge_dropout_probability"][edge_dropout_probability_index]
        hidden_dimension = HP_SEARCH_SPACE_Mapping["hidden_dimension"][hidden_dimension_index]

        print("The Optimal learning_rate: %f / "
              "weight_decay: %f / "
              "node_element_dropout_probability: %f /"
              "edge_dropout_probability: %f / "
              "embedding_dim: %f" % (
               learning_rate,
               weight_decay,
               node_element_dropout_probability,
               edge_dropout_probability,
               hidden_dimension))

        return learning_rate, weight_decay, node_element_dropout_probability, edge_dropout_probability, hidden_dimension

if __name__=="__main__":

    t1 = time.time()
    gnn_architecture = ["GCNConv", "GraphNorm", "Elu", "GCNConv", "GraphNorm", "Elu"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_name = "CS"
    graph = GraphData(data_name, shuffle=False).data
    information = "CS_GraphNAS"
    HS = HpSearchObj(gnn_architecture=gnn_architecture,
                     gnn_train_epoch=10,
                     graph=graph,
                     device=device,
                     information=information)

    HS.hp_tuning(10)
    t = time.time()
    print("Time Cost:", t-t1)
