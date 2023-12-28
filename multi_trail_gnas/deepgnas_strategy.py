import torch
import random
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from planetoid import GraphData
from torch_geometric.loader import ClusterData, ClusterLoader
from multi_trail_gnas.multi_trail_evaluation import MultiTrailEvaluation
from multi_trail_gnas.discrete_search_space import gnn_topology, component_candidate_dict

# 定义Q网络
class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim):
        super(QNetwork, self).__init__()
        self.layer_1 = nn.Linear(state_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x

class QLearningAgent(object):

    def __init__(self,
                 gamma,
                 epsilon,
                 test_epsilon,
                 agent_train_epoch,
                 hidden_dim,
                 state_space_vec_dim,
                 action_space_vec_dim):

        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 探索率
        self.agent_train_epoch = agent_train_epoch
        self.hidden_dim = hidden_dim
        self.state_space_vec_dim = state_space_vec_dim
        self.action_space_vec_dim = action_space_vec_dim
        self.parameters_list = []

        # 初始化 q network 与 target network
        self.q_network = QNetwork(self.state_space_vec_dim,
                                  self.action_space_vec_dim,
                                  self.hidden_dim)

        self.target_network = QNetwork(self.state_space_vec_dim,
                                       self.action_space_vec_dim,
                                       self.hidden_dim)

        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.parameters_list.append({"params": self.q_network.parameters()})

        # 基于GNN结构组件的所有候选项,构建环境状态反馈空间embedding,
        # 输入动作编号反馈执行此动作达到的状态空间向量.
        self.all_candidate_list = []
        self.env_state_index_list = []

        for key in component_candidate_dict:
            self.all_candidate_list = self.all_candidate_list + component_candidate_dict[key]
            self.env_state_index_list.append(len(component_candidate_dict[key]))
        self.env_state_feedback = torch.nn.Embedding(len(self.all_candidate_list),
                                                     self.state_space_vec_dim)
        self.parameters_list.append({"params": self.env_state_feedback.parameters()})

        # 基于GNN不同的组件候选作空间，构建不同动作空间向量解码器，
        # 输入不同动作空间向量，解码输出此动作空间中的具体 1.动作(candidates) / 2.动作编号;
        # 动作编号的作用 1.从环境中获取next state;
        #              2.使用replay buffer回收,构建一个强化学习训练样本.
        self.component_candidate_dict = component_candidate_dict
        self.decoders = torch.nn.ModuleDict()
        for component in self.component_candidate_dict:
            size = len(self.component_candidate_dict[component])
            decoder = torch.nn.Linear(self.action_space_vec_dim, size)
            self.decoders[component] = decoder
            self.parameters_list.append({"params": decoder.parameters()})

        # 初始优化器与损失函数
        self.optimizer = optim.Adam(self.parameters_list)
        self.loss_fn = nn.MSELoss()
        self.decay_epsilon_list = self.cosine_decay(epsilon=self.epsilon,
                                                    agent_train_epoch=self.agent_train_epoch)
        self.test_epsilon = test_epsilon

    def select_action(self,
                      state,
                      action_space_index,
                      train_epoch,
                      training=True):

        # 选择最后一个GNN组件，本次agent与环境交互结束，done标志置1，
        # 为计算最后的next_q做准备
        if action_space_index == 5:
            done = 1
        else:
            done = 0

        if action_space_index > 2:
            action_space_index -= 3

        # 基于动作空间索引,获取环境状态反馈索引
        if sum(self.env_state_index_list[:action_space_index]) == 0:
            state_index = sum(self.env_state_index_list[:action_space_index])
        else:
            state_index = sum(self.env_state_index_list[:action_space_index]) - 1

        # 获取动作空间
        component = gnn_topology[action_space_index]
        component_candidates = component_candidate_dict[component]

        if training:
            # agent 训练时动作采样有1.探索2.利用,两种情况且探索率从1按照余弦衰减退化到0
            if random.random() < self.decay_epsilon_list[train_epoch]:
                print("Explore Action")
                action_space_vec = torch.randn(1, self.action_space_vec_dim)
            else:
                print("Exploit Action")
                action_space_vec = self.q_network(state)
        else:
            # agent 测试阶段仍然有1.探索2.利用,两种情况,此时探索率为一个常数
            if random.random() < self.test_epsilon:
                action_space_vec = torch.randn(1, self.action_space_vec_dim)
            else:
                action_space_vec = self.q_network(state)

        action_probability_vec = self.decoders[component](action_space_vec)
        action_id = torch.argmax(action_probability_vec).item()
        candidate = component_candidates[action_id]

        return action_id, state_index, candidate, done

    def train(self, replay):

        for sample in replay:

            state = sample[0]
            action_space_index = sample[1]
            action_id = sample[2]
            next_state = sample[3]
            done = sample[4]
            reward = sample[5]

            action_space_vec = self.q_network(state)
            component = gnn_topology[action_space_index]
            action_probability_vec = self.decoders[component](action_space_vec)
            q_value = action_probability_vec[0][action_id]

            next_q_value = self.target_network(next_state).max()
            expected_q_values = reward + self.gamma * next_q_value * (1-done)

            loss = self.loss_fn(q_value, expected_q_values.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Agent Loss:", loss.item())

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        print("Target Network Update Complete")

    def cosine_decay(self, epsilon, agent_train_epoch, decay_interval=0.000001):

        decay_epsilon_list = []

        for step in range(agent_train_epoch):
            decayed_value = epsilon * 0.5 * (1 + np.cos(np.pi * step / agent_train_epoch))
            decay_epsilon_list.append(decayed_value)
            if decayed_value < decay_interval:
                break
        return decay_epsilon_list


class DeepGNAS(object):

    def __init__(self,
                 estimator,
                 search_parameter):

        self.estimator = estimator
        self.gamma = search_parameter["gamma"]  # 折扣因子
        self.epsilon = search_parameter["epsilon"]  # 探索率
        self.test_epsilon = search_parameter["test_epsilon"]
        self.hidden_dim = search_parameter["hidden_dim"]
        self.agent_train_epoch = search_parameter["agent_train_epoch"]
        self.state_space_vec_dim = search_parameter["state_space_vec_dim"]
        self.action_space_vec_dim = search_parameter["action_space_vec_dim"]
        self.target_network_update_epoch = search_parameter["target_network_update_epoch"]
        self.agent = QLearningAgent(self.gamma,
                                    self.epsilon,
                                    self.test_epsilon,
                                    self.agent_train_epoch,
                                    self.hidden_dim,
                                    self.state_space_vec_dim,
                                    self.action_space_vec_dim)

    def search(self,
               agent_train_epoch,
               scale_of_sampled_gnn,
               return_top_k):

        # agent training
        best_reward = 0
        for epoch in range(agent_train_epoch):

            print("Agent Training Epoch", epoch)
            state = torch.randn(1, self.state_space_vec_dim)
            action_space_index = 0
            gnn_architecture = []
            replay = []

            # 与环境开始交互
            for _ in range(len(gnn_topology)):

                action_id, state_index, candidate, done = self.agent.select_action(state, action_space_index, epoch, training=True)
                next_state = self.agent.env_state_feedback(torch.tensor([action_id + state_index + 1]))
                replay.append([state, action_space_index, action_id, next_state, done])
                gnn_architecture.append(candidate)
                action_space_index += 1
                state = next_state

            reward = self.estimator.get_estimation_score(gnn_architecture)
            print("Sampled GNN Architecture In Agent Training:", gnn_architecture, "Reward:", reward)

            # reshape reward 构造
            reshape_reward = reward / len(gnn_topology)
            for index in range(len(replay)):
                replay[index] = replay[index] + [reshape_reward]

            if reward > best_reward:
                best_reward = reward
            print("Best Reward:", best_reward)

            # 基于采样一个完整的GNN动作行为训练agent
            self.agent.train(replay)

            # 更新 target network 参数
            if epoch % self.target_network_update_epoch == 0:
                self.agent.update_target_network()
                print(f"Epoch: {epoch}, Best Reward: {best_reward}")

        print("Agent Training Complete And Sampling Promising GNN Architectures For HPO")
        gnn_architecture_list = []
        gnn_architecture_val_score = []

        for epoch in range(scale_of_sampled_gnn):

            state = torch.randn(1, self.state_space_vec_dim)
            action_space_index = 0
            gnn_architecture = []

            for _ in range(len(gnn_topology)):

                action_id, state_index, candidate, _ = self.agent.select_action(state, action_space_index, _, training=False)
                next_state = self.agent.env_state_feedback(torch.tensor([action_id + state_index + 1]))
                gnn_architecture.append(candidate)
                state = next_state
                action_space_index += 1

            score = self.estimator.get_estimation_score(gnn_architecture)
            gnn_architecture_list.append(gnn_architecture)
            gnn_architecture_val_score.append(score)

        print(gnn_architecture_list)
        print(gnn_architecture_val_score)

        top_gnn, top_gnn_performance = self.estimator.rank_based_estimation_score(gnn_architecture_list,
                                                                                  gnn_architecture_val_score,
                                                                                  return_top_k)
        print(top_gnn)
        print(top_gnn_performance)

        for gnn_, val_score_ in zip(top_gnn, top_gnn_performance):
            print(gnn_, val_score_)

        return top_gnn, top_gnn_performance


if __name__ == "__main__":

    data_name = "CS"
    graph = GraphData(data_name, shuffle=False).data
    cluster_data = ClusterData(graph, num_parts=1)
    graph_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gnn_model_config = {"num_node_features": graph.num_node_features,
                        "num_classes": graph.num_classes,
                        "hidden_dimension": 128,
                        "learn_rate": 0.0001,
                        "node_element_dropout_probability": 0.6,
                        "edge_dropout_probability": 0.3,
                        "weight_decay": 0.0001,
                        "train_epoch": 5}

    estimator = MultiTrailEvaluation(gnn_model_config=gnn_model_config,
                                     graph=graph_loader,
                                     device=device)
    agent_train_epoch = 10

    search_parameter = {"gamma": 1.0,
                        "epsilon": 1.0,
                        "test_epsilon": 0.2,
                        "agent_train_epoch": agent_train_epoch,
                        "hidden_dim": 128,
                        "state_space_vec_dim": 128,
                        "action_space_vec_dim": 128,
                        "target_network_update_epoch": 2}

    searcher = DeepGNAS(estimator=estimator,
                        search_parameter=search_parameter)

    top_gnn, top_gnn_score = searcher.search(agent_train_epoch=agent_train_epoch,
                                             scale_of_sampled_gnn=5,
                                             return_top_k=2)