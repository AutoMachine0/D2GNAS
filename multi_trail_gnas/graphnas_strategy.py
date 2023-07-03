import torch
import numpy as np
import scipy.signal
import torch.nn.functional as F
from planetoid import GraphData
from torch.autograd import Variable
from torch_geometric.loader import ClusterData, ClusterLoader
from multi_trail_gnas.multi_trail_evaluation import MultiTrailEvaluation
from multi_trail_gnas.discrete_search_space import gnn_topology, component_candidate_dict

class Controller(torch.nn.Module):

    def __init__(self,
                 controller_hidden_dimension,
                 device=None):

        super(Controller, self).__init__()

        self.controller_hidden_dimension = controller_hidden_dimension
        self.gnn_topology = gnn_topology
        self.component_candidate_dict = component_candidate_dict

        if device:
            self.device = device
        else:
            self.device = "cpu"

        self.all_candidate_list = []
        self.each_component_candidate_num_list = []

        for key in self.component_candidate_dict:

            self.all_candidate_list = self.all_candidate_list + self.component_candidate_dict[key]
            self.each_component_candidate_num_list.append(len(self.component_candidate_dict[key]))

        # build all candidate operators learnable embedding
        self.encoder = torch.nn.Embedding(len(self.all_candidate_list),
                                          self.controller_hidden_dimension)

        # build series model
        self.lstm = torch.nn.LSTMCell(self.controller_hidden_dimension,
                                      self.controller_hidden_dimension)

        # build mlp decoder for different components
        self.decoders = torch.nn.ModuleDict()
        for component in self.component_candidate_dict:
            size = len(self.component_candidate_dict[component])
            decoder = torch.nn.Linear(self.controller_hidden_dimension, size)
            self.decoders[component] = decoder

        self.reset_parameters()

    def construct_action(self, actions):
        structure_list = []
        for single_action in actions:
            structure = []
            for action, action_name in zip(single_action, self.gnn_topology):
                predicted_actions = self.component_candidate_dict[action_name][action]
                structure.append(predicted_actions)
            structure_list.append(structure)
        return structure_list

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self.decoders:
            self.decoders[decoder].bias.data.fill_(0)

    def forward(self,
                inputs,
                hidden,
                component_name):

        hx, cx = self.lstm(inputs, hidden)
        logits = torch.relu(self.decoders[component_name](hx))

        return logits, (hx, cx)

    def component_index(self, component_name):
        gnn_component_names = self.component_candidate_dict.keys()
        for i, name in enumerate(gnn_component_names):
            if component_name == name:
                return i

    def get_variable(self, inputs, cuda=False, **kwargs):
        if type(inputs) in [list, np.ndarray]:
            inputs = torch.Tensor(inputs)
        if cuda:
            out = Variable(inputs.cuda(), **kwargs)
        else:
            out = Variable(inputs, **kwargs)
        return out

    def sample(self, num_sampled_gnn_architecture=1):

        if num_sampled_gnn_architecture < 1:
            raise Exception(f'Wrong Number of Sampled GNN Architecture: {num_sampled_gnn_architecture} < 1')

        inputs = torch.zeros([num_sampled_gnn_architecture, self.controller_hidden_dimension]).to(self.device)
        hidden = (torch.zeros([num_sampled_gnn_architecture, self.controller_hidden_dimension]).to(self.device),
                  torch.zeros([num_sampled_gnn_architecture, self.controller_hidden_dimension]).to(self.device))

        entropies = []
        log_probs = []
        actions = []

        for gnn_component in self.gnn_topology:

            decoder_index = self.component_index(gnn_component)

            logits, hidden = self.forward(inputs,
                                          hidden,
                                          gnn_component)

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)
            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(1, self.get_variable(action, requires_grad=False))
            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            inputs = self.get_variable(
                action[:, 0] + sum(self.each_component_candidate_num_list[:decoder_index]),
                True,
                requires_grad=False)

            inputs = self.encoder(inputs)
            actions.append(action[:, 0])

        actions = torch.stack(actions).transpose(0, 1)
        dags = self.construct_action(actions)

        return dags, torch.cat(log_probs), torch.cat(entropies)

class GraphNAS(object):

    def __init__(self,
                 estimator,
                 controller_parameter,
                 device):

        self.estimator = estimator
        self.history_gnn = []
        self.history_gnn_val_score = []
        self.history = []
        self.device = device
        self.move_average_reward_operator = MoveAverageOperator()

        if "controller_hidden_dimension" not in controller_parameter.keys():
            controller_hidden_dimension = 100
        else:
            controller_hidden_dimension = controller_parameter["controller_hidden_dimension"]

        if "controller_optimizer_learn_rate" not in controller_parameter.keys():
            controller_optimizer_learn_rate = 0.001
        else:
            controller_optimizer_learn_rate = controller_parameter["controller_optimizer_learn_rate"]

        if "controller_optimizer_weight_decay" not in controller_parameter.keys():
            controller_optimizer_weight_decay = 0.0005
        else:
            controller_optimizer_weight_decay = controller_parameter["controller_optimizer_weight_decay"]

        self.search_controller = Controller(controller_hidden_dimension, self.device).to(self.device)
        print("Controller Building Finish")

        # build search controller optimizer
        self.controller_optim = torch.optim.Adam(self.search_controller.parameters(),
                                                 lr=controller_optimizer_learn_rate,
                                                 weight_decay=controller_optimizer_weight_decay)

    def search(self,
               controller_train_epoch=10,
               num_sampled_gnn_for_one_train_epoch=1,
               scale_of_sampled_gnn=10,
               return_top_k=10):

        print("Reinforcement Learning Search Starting")
        print(64 * "=")
        print("Step 1 : Controller Training")
        print("Number of Sampled GNN Architecture for One Training Epoch is",
              num_sampled_gnn_for_one_train_epoch)

        for epoch in range(controller_train_epoch):
            print(32*"+")
            print("Training Epoch:", epoch+1)
            self.search_controller_train(num_sampled_gnn_for_one_train_epoch)

        print(64 * "=")
        print("Step 2 : Controller Sampling")
        print("The Scale of Sampled GNN Architectures with Trained Controller is",
              scale_of_sampled_gnn)

        top_gnn, top_val_score = self.sample_gnn_with_trained_controller(scale_of_sampled_gnn, return_top_k)

        print(64 * "=")
        print("Multi-trail GraphNAS Search Strategy Completion")
        return top_gnn, top_val_score

    def search_controller_train(self, num_sampled_gnn_for_one_train_epoch):

        self.search_controller.train()
        gnn_architecture_list = []
        log_probs_list = []
        entropies_list = []

        for _ in range(num_sampled_gnn_for_one_train_epoch):
            gnn_architecture, log_probs, entropies = self.search_controller.sample()
            np_entropies = entropies.data.cpu().numpy()
            gnn_architecture_list.append(gnn_architecture[0])
            log_probs_list.append(log_probs)
            entropies_list.append(np_entropies)

        # get multiple reward list as rewards_list based on multiple GNN architectures
        rewards_list = self.get_reward(gnn_architecture_list, entropies_list)
        torch.cuda.empty_cache()

        # get discount rewards_list
        discount_rewards_list = []
        for index in range(len(rewards_list)):
            discount_rewards_list.append(self.discount(rewards_list[index], 1.0))
        rewards_list = discount_rewards_list

        for gnn_architecture, discount_reward in zip(gnn_architecture_list, discount_rewards_list):
            print("Sampled GNN Architecture:", gnn_architecture)
            print("    Discount Reward List:", discount_reward)

        # get moving average baseline
        # select max reward vector index from rewards_list
        max_index = self.select_max_reward_vector_index(rewards_list)
        baseline = rewards_list[max_index]
        adv_list = []

        for rewards in rewards_list:
            adv_list.append(rewards - baseline)
        self.history.append(rewards_list[max_index])

        # accumulation a adv_list gradient for optimizer
        for index in range(len(adv_list)):
            adv = self.scale(adv_list[index], scale_value=0.5)
            adv = self.get_variable(adv, self.device, requires_grad=False)
            # calculate every reward policy loss
            loss = -log_probs_list[index] * adv
            loss = loss.sum()
            # calculate gradient based on loss backward
            loss.backward()

        # update controller parameter based on optimizer
        self.controller_optim.step()
        self.controller_optim.zero_grad()

        torch.cuda.empty_cache()

    def get_variable(self, inputs, cuda=False, **kwargs):
        if type(inputs) in [list, np.ndarray]:
            inputs = torch.Tensor(inputs)
        if cuda:
            out = Variable(inputs.cuda(), **kwargs)
        else:
            out = Variable(inputs, **kwargs)
        return out

    def scale(self, value, scale_value=1.0, last_k=10):
        # scale value into [-scale_value, scale_value], according last_k history
        # find the large number in multiple lists
        max_reward = np.max(self.history[-last_k:])
        if max_reward == 0:
            return value
        return scale_value / max_reward * value

    def discount(self, x, amount):
        return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]

    def get_reward(self, gnn_architecture_list, entropies_list):
        # compute multiple rewards as list based on multiple sampled models on validation data.
        val_score_list = []
        for gnn_architecture in gnn_architecture_list:
            val_score = self.estimator.get_estimation_score(gnn_architecture)
            val_score_list.append(val_score)

        # get multiple reward based on move average strategy
        move_average_reward_list = []
        for val_acc in val_score_list:
            move_average_reward_list.append(self.move_average_reward_operator.get_reward(val_acc))

        rewards_list = []
        for index in range(len(move_average_reward_list)):
            reward_list = []
            reward_list.append(move_average_reward_list[index])
            rewards_list.append(reward_list + 1e-4 * entropies_list[index])

        return rewards_list

    def select_max_reward_vector_index(self, rewards_list):
        max_index = None
        max_reward_vector = None
        for index in range(len(rewards_list)):
            if index == 0:
                max_reward_vector = np.sum(rewards_list[index])
                max_index = index
            elif max_reward_vector < np.sum(rewards_list[index]):
                max_reward_vector = np.sum(rewards_list[index])
                max_index = index
        return max_index

    def sample_gnn_with_trained_controller(self, scale_sampled_gnn, return_top_k):

        val_score_list = []
        gnn_architecture_list = []

        for _ in range(scale_sampled_gnn):
            gnn_architecture, _, _ = self.search_controller.sample()
            gnn_architecture_list.append(gnn_architecture[0])

        # sampled gnn architecture estimation
        for gnn_architecture in gnn_architecture_list:
            val_score = self.estimator.get_estimation_score(gnn_architecture)
            val_score_list.append(val_score)

        # rank sampled gnn architecture
        ranked_gnn_architecture_list, ranked_val_score_list = self.estimator.rank_based_estimation_score(gnn_architecture_list,
                                                                                                         val_score_list,
                                                                                                         return_top_k)
        print("Sampled Top", return_top_k, "GNN Architecture and Estimation Score (Validation Accuracy):")
        for gnn, val_score in zip(ranked_gnn_architecture_list, ranked_val_score_list):
            print(gnn, val_score)

        return ranked_gnn_architecture_list, ranked_val_score_list

class MoveAverageOperator(object):

    def __init__(self, top_k=10):
        self.scores = []
        self.top_k = top_k

    def get_average(self, score):
        if len(self.scores) > 0:
            avg = np.mean(self.scores)
        else:
            avg = 0

        self.scores.append(score)
        self.scores.sort(reverse=True)
        self.scores = self.scores[:self.top_k]
        return avg

    def get_reward(self, score):
        reward = score - self.get_average(score)
        return np.clip(reward, -0.5, 0.5)

if __name__=="__main__":

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
                        "train_epoch": 10}

    estimator = MultiTrailEvaluation(gnn_model_config=gnn_model_config,
                                     graph=graph_loader,
                                     device=device)

    search_parameter = {"controller_hidden_dimension": 100,
                        "controller_optimizer_learn_rate": 0.0001,
                        "controller_optimizer_weight_decay": 0.0005}

    searcher = GraphNAS(estimator,
                        search_parameter,
                        device)

    searcher.search(controller_train_epoch=10,
                    num_sampled_gnn_for_one_train_epoch=1,
                    scale_of_sampled_gnn=5,
                    return_top_k=2)
