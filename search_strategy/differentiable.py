import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArchitectureGradientOptimizer(torch.nn.Module):
    def __init__(self, supernet, lr, decay, device):

        super(ArchitectureGradientOptimizer, self).__init__()

        # build learnable architecture alpha parameter based on supernet.
        self.architecture_alpha_list = []

        for component, candidates in supernet.component_candidate_dict.items():

            architecture_alpha = Variable(torch.Tensor(supernet.num_gnn_layer, len(candidates))).to(device)
            architecture_alpha.requires_grad = True
            nn.init.uniform_(architecture_alpha)

            self.architecture_alpha_list.append(architecture_alpha)

        # optimizer for learnable architecture alpha parameter
        self.optimizer = torch.optim.Adam(self.architecture_alpha_list,
                                          lr=lr,
                                          weight_decay=decay)

        # loss function for learnable architecture alpha parameter
        self.loss = torch.nn.CrossEntropyLoss()
    def sample_gnn_model_build(self,  sample_architecture, supernet):
        """
        passing current supernet and construct the gnn model based the gnn architecture sampled by gumbel softmax
        """

        self.pre_process_mlp = supernet.pre_process_mlp
        self.post_process_mlp = supernet.post_process_mlp

        self.layer1_conv = supernet.supernet_operation_pool[0].get_candidate(sample_architecture[0])
        self.layer1_norm = supernet.supernet_operation_pool[1].get_candidate(sample_architecture[1])
        self.layer1_act = supernet.supernet_operation_pool[2].get_candidate(sample_architecture[2])

        self.layer2_conv = supernet.supernet_operation_pool[3].get_candidate(sample_architecture[3])
        self.layer2_norm = supernet.supernet_operation_pool[4].get_candidate(sample_architecture[4])
        self.layer2_act = supernet.supernet_operation_pool[5].get_candidate(sample_architecture[5])
    def forward(self,
                x,
                edge_index,
                gumbel_softmax_sample_ret_list,
                sample_candidate_index_list):

        # add architecture alpha parameter to Torch Computation Graph of sampled gnn model for getting validation gradient
        x = self.pre_process_mlp(x)
        x = self.layer1_conv(x, edge_index) * gumbel_softmax_sample_ret_list[0][0][sample_candidate_index_list[0]]
        x = self.layer1_norm(x) * gumbel_softmax_sample_ret_list[1][0][sample_candidate_index_list[1]]
        x = self.layer1_act(x) * gumbel_softmax_sample_ret_list[2][0][sample_candidate_index_list[2]]

        x = self.layer2_conv(x, edge_index) * gumbel_softmax_sample_ret_list[0][1][sample_candidate_index_list[3]]
        x = self.layer2_norm(x) * gumbel_softmax_sample_ret_list[1][1][sample_candidate_index_list[4]]
        x = self.layer2_act(x) * gumbel_softmax_sample_ret_list[2][1][sample_candidate_index_list[5]]

        x = self.post_process_mlp(x)

        return x

class DifferentiableSearch(object):
    def __init__(self,
                 supernet,
                 graph,
                 differentiable_search_optimizer_config_dict,
                 temperature,
                 device=None):

        if device:
            self.device = device
        else:
            self.device = device_

        self.differentiable_search_pruning_best_gnn_history = []
        self.graph = graph
        self.temperature = temperature
        self.best_architecture_history = []
        self.architecture_gradient_optimizer = ArchitectureGradientOptimizer(supernet,
                                                                             differentiable_search_optimizer_config_dict["lr"],
                                                                             differentiable_search_optimizer_config_dict["decay"],
                                                                             self.device)

    def search(self,
               supernet,
               search_paths,
               search_epoch,
               return_top_k):

        print("Differentiable Search Starting")
        differentiable_pruning_search_path = []

        # get the architecture alpha parameter sample distribution for gumbel softmax sample
        architecture_alpha_list = self.architecture_gradient_optimizer.architecture_alpha_list

        for epoch in range(search_epoch):
            print(32 * "=")
            print("Search Epoch:", epoch+1)
            gumbel_softmax_sample_output_list = []

            for architecture_alpha in architecture_alpha_list:
                gumbel_softmax_sample_output_list.append(self.hard_gumbel_softmax_sample(F.softmax(architecture_alpha, dim=-1)))

            # pruning search space sample constraint
            re_sample, \
            sample_candidate_index_list, \
            sample_architecture = self.sample_gnn_architecture_check(gumbel_softmax_sample_output_list,
                                                                     supernet,
                                                                     search_paths)

            while re_sample:

                gumbel_softmax_sample_output_list = []

                for architecture_alpha in architecture_alpha_list:
                    gumbel_softmax_sample_output_list.append(self.hard_gumbel_softmax_sample(F.softmax(architecture_alpha, dim=-1)))

                re_sample, \
                sample_candidate_index_list, \
                sample_architecture = self.sample_gnn_architecture_check(gumbel_softmax_sample_output_list,
                                                                         supernet,
                                                                         search_paths)

            # save the gnn architecture searched by differentiable search
            differentiable_pruning_search_path.append(sample_architecture)

            # architecture alpha parameter optimization based on the sampled gnn model using the validation gradient
            self.architecture_gradient_optimizer.train()
            self.architecture_gradient_optimizer.sample_gnn_model_build(sample_architecture, supernet)

            # architecture alpha parameter optimization
            y_pred = self.architecture_gradient_optimizer(self.graph.x,
                                                          self.graph.edge_index,
                                                          gumbel_softmax_sample_output_list,
                                                          sample_candidate_index_list)

            loss = self.architecture_gradient_optimizer.loss(y_pred[self.graph.val_mask],
                                                             self.graph.y[self.graph.val_mask])

            self.architecture_gradient_optimizer.optimizer.zero_grad()
            loss.backward()
            self.architecture_gradient_optimizer.optimizer.step()

            best_gnn = self.best_alpha_gnn_architecture(self.architecture_gradient_optimizer.architecture_alpha_list,
                                                        supernet)

            print("Best GNN Architecture:", best_gnn)

        print(32 * "=")

        print("differentiable Search Ending")

        if int(return_top_k) <= len(self.best_architecture_history):
            best_alpha_gnn_architecture_list = self.best_architecture_history[-int(return_top_k):]
        else:
            best_alpha_gnn_architecture_list = self.best_architecture_history

        for gnn in best_alpha_gnn_architecture_list:
            if gnn not in self.differentiable_search_pruning_best_gnn_history:
                self.differentiable_search_pruning_best_gnn_history.append(gnn)

        if len(self.differentiable_search_pruning_best_gnn_history) > int(return_top_k):
            self.differentiable_search_pruning_best_gnn_history = self.differentiable_search_pruning_best_gnn_history[-return_top_k:]

        print("differentiable Search Final Output Best GNN Architectures:")

        for gnn in self.differentiable_search_pruning_best_gnn_history:
            print(gnn)
        
        return self.differentiable_search_pruning_best_gnn_history, differentiable_pruning_search_path
    def hard_gumbel_softmax_sample(self, sample_probability):

        hard_gumbel_softmax_sample_output = F.gumbel_softmax(logits=sample_probability,
                                                             tau=self.temperature,
                                                             hard=True)
        return hard_gumbel_softmax_sample_output
    def sample_gnn_architecture_check(self, gumbel_softmax_sample_ret_list, supernet, search_paths):

        candidate_list = []
        candidate_index_list = []

        for component_one_hot, component in zip(gumbel_softmax_sample_ret_list, supernet.component_candidate_dict.keys()):

            for candidate_one_hot in component_one_hot.cpu().detach().numpy().tolist():
                candidate_index = candidate_one_hot.index(max(candidate_one_hot))
                candidate_list.append(supernet.component_candidate_dict[component][candidate_index])
                candidate_index_list.append(candidate_index)

        sample_architecture = candidate_list[::2] + candidate_list[1::2]
        sample_candidate_index = candidate_index_list[::2] + candidate_index_list[1::2]

        if sample_architecture not in search_paths:
            re_sample = True
        else:
            print("Gumbel Softmax Sample GNN Architecture:", sample_architecture)
            re_sample = False

        return re_sample, sample_candidate_index, sample_architecture
    def best_alpha_gnn_architecture(self, architecture_alpha_list, supernet):

        best_alpha_architecture = []

        for architecture_alpha_vector_list, component in zip(architecture_alpha_list,
                                                             supernet.component_candidate_dict.keys()):

            for architecture_alpha_vector in architecture_alpha_vector_list.cpu().detach().numpy().tolist():
                best_alpha_index = architecture_alpha_vector.index(max(architecture_alpha_vector))
                best_alpha_architecture.append(supernet.component_candidate_dict[component][best_alpha_index])

        gnn_architecture = best_alpha_architecture[::2] + best_alpha_architecture[1::2]

        if gnn_architecture not in self.best_architecture_history:
            self.best_architecture_history.append(gnn_architecture)

        return gnn_architecture

if __name__=="__main__":

    pass
