import os
import torch
import numpy as np
from scalable_gnn.metric import accuracy
from planetoid import GraphData
from scalable_gnn.message_upd import MLPUpdator
from scalable_gnn.pre_pro import pre_processing
from scalable_gnn.post_pro import post_processing
from scalable_gnn.message_agg import MessageAdaAgg
from scalable_gnn.base_op import LossFunction, OptimationFunction, x_row_normalization
from scalable_gnn.graph_agg import Augmented_Normalized_Adjacency, Personalized_Pagerank

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PaScaV3():

    def __init__(self,
                 feature_dim,
                 num_classes,
                 dropout_rate,
                 hidden_dim,
                 learning_rate,
                 weight_decay,
                 norm_name):

        super(PaScaV3, self).__init__()

        self.pre_graph_aggregator = Augmented_Normalized_Adjacency
        self.message_aggregator = MessageAdaAgg(feature_dim).to(device)
        self.message_updator = MLPUpdator(input_dim=feature_dim,
                                          output_dim=num_classes,
                                          dropout_rate=dropout_rate,
                                          hidden_dim_list=[hidden_dim, hidden_dim],
                                          act_list=["relu", "relu"],
                                          norm_name=norm_name).to(device)

        self.post_graph_aggregator = Personalized_Pagerank

        self.loss_f = LossFunction().loss_get("crossentropy")

        model_parameter = [{"params": self.message_aggregator.parameters()},
                           {"params": self.message_updator.parameters()}]

        self.optimizer = OptimationFunction().optimation_get("adam")(model_parameter,
                                                                     lr=learning_rate,
                                                                     weight_decay=weight_decay)

    def train(self, x, A, y, train_epoch, train_mask, test_mask):

        # pre-processing
        M = pre_processing(x=x,
                           A=A,
                           k=6,
                           graph_aggregator=self.pre_graph_aggregator)

        best_test = 0
        for epoch in range(train_epoch):
            # training
            self.message_aggregator.train()
            self.message_updator.train()
            c = self.message_aggregator(M)
            h = self.message_updator(c)

            loss = self.loss_f(h[train_mask], y[train_mask])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # testing
            self.message_aggregator.eval()
            self.message_updator.eval()
            # post-processing
            y_pred = post_processing(x=h,
                                     A=A,
                                     k=4,
                                     graph_aggregator=self.post_graph_aggregator,
                                     alpha=0.3)

            test_accuracy = accuracy(y_pred[test_mask], y[test_mask])

            if test_accuracy > best_test:
                best_test = test_accuracy

        return best_test

def edge_index_to_sparse_adj(x, edge_index):

    number_nodes = x.size()[0]
    adj = torch.zeros(number_nodes, number_nodes)

    for index in range(len(edge_index[0])):
        adj[edge_index[0][index], edge_index[1][index]] = 1

    return adj

if __name__=="__main__":

    test_iter = 1
    train_epoch = 100
    dropout_rate = 0.0
    hidden_dim = 128
    learning_rate = 0.05
    weight_decay = 0.0005
    norm_name = "GraphNorm"
    data_list = ["CS", "Photo", "Pubmed", "Physics", "Computers"]
    for data in data_list:
        test_acc_list = []
        for iter in range(test_iter):
            graph = GraphData(data, shuffle=False).data
            x = graph.x
            y = graph.y
            train_mask = graph.train_mask
            test_mask = graph.test_mask
            feature_dim = graph.num_node_features
            num_classes = graph.num_classes

            A = edge_index_to_sparse_adj(x, graph.edge_index)
            A = A.numpy()
            x = x_row_normalization(x)
            x = x.to('cpu').numpy()

            pasca3 = PaScaV3(feature_dim,
                             num_classes,
                             dropout_rate=dropout_rate,
                             hidden_dim=hidden_dim,
                             learning_rate=learning_rate,
                             weight_decay=weight_decay,
                             norm_name=norm_name)

            test_acc = pasca3.train(x=x,
                                    A=A,
                                    y=y,
                                    train_epoch=train_epoch,
                                    train_mask=train_mask,
                                    test_mask=test_mask)
            test_acc_list.append(test_acc)
            print(str(iter+1)+" Test Epoch: "+str(test_acc))

        avg_test = np.array(test_acc_list).mean()
        std_test = np.array(test_acc_list).std()

        dir = os.path.abspath(__file__)[:-len(os.path.basename(__file__))] + "Performance/"

        if not os.path.exists(dir):
            os.makedirs(dir)

        path = dir + "PaScaV3" + "_" + data + ".txt"

        with open(path, "a+") as f:
            f.write("avg test accuracy: " + str(avg_test) + " std test accuracy:" + str(std_test) + "\n")