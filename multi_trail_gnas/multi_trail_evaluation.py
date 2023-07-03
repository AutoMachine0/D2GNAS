import torch
import numpy as np
import torch.nn.functional as F
from search_space.mlp import MLP
from search_space.act_pool import ActPool
from search_space.conv_pool import ConvPool
from search_space.norm_pool import NormPool
from graph_augment.edge_dropout_pyg import dropout_edge

class GNNBuildWithArchitecture(torch.nn.Module):

    def __init__(self,
                 num_node_features,
                 num_classes,
                 hidden_dimension,
                 node_element_dropout_probability,
                 edge_dropout_probability,
                 architecture):

        super(GNNBuildWithArchitecture, self).__init__()

        self.layer1_act_pool = ActPool()
        self.layer2_act_pool = ActPool()

        self.node_element_dropout_probability = node_element_dropout_probability
        self.edge_dropout_probability = edge_dropout_probability

        # build new gnn model based on gnn architecture
        self.pre_process_mlp = MLP(input_dim=num_node_features,
                                   output_dim=hidden_dimension)

        self.post_process_mlp = MLP(input_dim=hidden_dimension,
                                    output_dim=num_classes)

        self.layer1_conv = ConvPool(hidden_dimension, hidden_dimension).get_conv(architecture[0])
        self.layer1_norm = NormPool(hidden_dimension).get_norm(architecture[1])
        self.layer1_act = self.layer1_act_pool.get_act(architecture[2])

        self.layer2_conv = ConvPool(hidden_dimension, hidden_dimension).get_conv(architecture[3])
        self.layer2_norm = NormPool(hidden_dimension).get_norm(architecture[4])
        self.layer2_act = self.layer2_act_pool.get_act(architecture[5])

    def forward(self, x, edge_index):

        x = self.pre_process_mlp(x)

        drop_edge_index = dropout_edge(edge_index=edge_index, p=self.edge_dropout_probability)[0]
        drop_x = F.dropout(x, p=self.node_element_dropout_probability)
        x = self.layer1_conv(drop_x, drop_edge_index)
        x = self.layer1_norm(x)
        x = self.layer1_act(x)

        drop_edge_index = dropout_edge(edge_index=edge_index, p=self.edge_dropout_probability)[0]
        drop_x = F.dropout(x, p=self.node_element_dropout_probability)
        x = self.layer2_conv(drop_x, drop_edge_index)
        x = self.layer2_norm(x)
        x = self.layer2_act(x)

        x = self.post_process_mlp(x)

        return x

class MultiTrailEvaluation(object):

    def __init__(self, gnn_model_config, graph, device):

        self.num_node_features = gnn_model_config["num_node_features"]
        self.num_classes = gnn_model_config["num_classes"]
        self.hidden_dimension = gnn_model_config["hidden_dimension"]
        self.node_element_dropout_probability = gnn_model_config["node_element_dropout_probability"]
        self.edge_dropout_probability = gnn_model_config["edge_dropout_probability"]
        self.learn_rate = gnn_model_config["learn_rate"]
        self.weight_decay = gnn_model_config["weight_decay"]
        self.train_epoch = gnn_model_config["train_epoch"]
        self.graph = graph
        self.batch_number = len(self.graph)
        self.device = device

    def get_estimation_score(self, architecture):

        # 构建GNN模型
        gnn_model = GNNBuildWithArchitecture(num_node_features=self.num_node_features,
                                             num_classes=self.num_classes,
                                             hidden_dimension=self.hidden_dimension,
                                             node_element_dropout_probability=self.node_element_dropout_probability,
                                             edge_dropout_probability=self.edge_dropout_probability,
                                             architecture=architecture).to(self.device)

        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     lr=self.learn_rate,
                                     weight_decay=self.weight_decay)

        loss_f = torch.nn.CrossEntropyLoss()

        # GNN模型训练
        gnn_model.train()
        for epoch in range(self.train_epoch):
            for sub_graph, step in zip(self.graph, range(self.batch_number)):

                y_pred = gnn_model(sub_graph.x, sub_graph.edge_index)

                loss = loss_f(y_pred[sub_graph.train_mask],
                              sub_graph.y[sub_graph.train_mask])/self.batch_number

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # GNN模型评估
        gnn_model.eval()
        val_iter = 0
        sum_val_acc = []

        for sub_graph in self.graph:

            y_pred = gnn_model(sub_graph.x, sub_graph.edge_index)
            pred = y_pred.argmax(dim=1)

            correct_val = pred[sub_graph.val_mask] == sub_graph.y[sub_graph.val_mask]
            if int(sub_graph.val_mask.sum()) > 0:
                sub_val_acc = int(correct_val.sum()) / int(sub_graph.val_mask.sum())
                sum_val_acc.append(sub_val_acc)
                val_iter += 1

        val_acc = np.array(sum_val_acc).sum() / val_iter

        return val_acc

    def get_best_validation_estimation(self, architecture):
        torch.cuda.empty_cache()
        # 构建GNN模型
        gnn_model = GNNBuildWithArchitecture(num_node_features=self.num_node_features,
                                             num_classes=self.num_classes,
                                             hidden_dimension=self.hidden_dimension,
                                             node_element_dropout_probability=self.node_element_dropout_probability,
                                             edge_dropout_probability=self.edge_dropout_probability,
                                             architecture=architecture).to(self.device)

        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     lr=self.learn_rate,
                                     weight_decay=self.weight_decay)

        loss_f = torch.nn.CrossEntropyLoss()

        best_val_acc = 0
        # GNN模型训练
        for epoch in range(self.train_epoch):

            gnn_model.train()
            for sub_graph, step in zip(self.graph, range(self.batch_number)):

                y_pred = gnn_model(sub_graph.x, sub_graph.edge_index)

                loss = loss_f(y_pred[sub_graph.train_mask],
                              sub_graph.y[sub_graph.train_mask])/self.batch_number

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # GNN模型评估
            gnn_model.eval()
            val_iter = 0
            sum_val_acc = []
            for sub_graph in self.graph:
                y_pred = gnn_model(sub_graph.x, sub_graph.edge_index)
                pred = y_pred.argmax(dim=1)

                correct_val = pred[sub_graph.val_mask] == sub_graph.y[sub_graph.val_mask]
                if int(sub_graph.val_mask.sum()) > 0:
                    sub_val_acc = int(correct_val.sum()) / int(sub_graph.val_mask.sum())
                    sum_val_acc.append(sub_val_acc)
                    val_iter += 1

            val_acc = np.array(sum_val_acc).sum() / val_iter

            if best_val_acc < val_acc:
                best_val_acc = val_acc

        return best_val_acc

    def get_test_score(self, architecture):
        # 构建GNN模型
        gnn_model = GNNBuildWithArchitecture(num_node_features=self.num_node_features,
                                             num_classes=self.num_classes,
                                             hidden_dimension=self.hidden_dimension,
                                             node_element_dropout_probability=self.node_element_dropout_probability,
                                             edge_dropout_probability=0.0,
                                             architecture=architecture).to(self.device)

        optimizer = torch.optim.Adam(gnn_model.parameters(),
                                     lr=self.learn_rate,
                                     weight_decay=self.weight_decay)

        loss_f = torch.nn.CrossEntropyLoss()

        best_test = 0
        for epoch in range(self.train_epoch):

            for sub_graph in self.graph:

                # GNN模型训练
                gnn_model.train()
                y_pred = gnn_model(sub_graph.x, sub_graph.edge_index)

                loss = loss_f(y_pred[sub_graph.train_mask],
                              sub_graph.y[sub_graph.train_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # GNN模型测试
                gnn_model.eval()
                y_pred = gnn_model(sub_graph.x, sub_graph.edge_index)
                pred = y_pred.argmax(dim=1)
                correct_test = pred[sub_graph.test_mask] == sub_graph.y[sub_graph.test_mask]
                test_acc = int(correct_test.sum()) / int(sub_graph.test_mask.sum())

                if test_acc > best_test:
                    best_test = test_acc

        return best_test

    def rank_based_estimation_score(self, gnn_list, val_score_list, top_k):

        gnn_dict = {}

        for key, value in zip(gnn_list, val_score_list):
            gnn_dict[str(key)] = value
        rank_gnn_dict = sorted(gnn_dict.items(), key=lambda x: x[1], reverse=True)

        rank_gnn = []
        rank_gnn_val_score = []

        i = 0
        for key, value in rank_gnn_dict:

            if i == top_k:
                break
            else:
                rank_gnn.append(eval(key))
                rank_gnn_val_score.append(value)
                i += 1
        return rank_gnn, rank_gnn_val_score