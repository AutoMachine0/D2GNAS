import copy
import torch
import random
import numpy as np
from planetoid import GraphData
from torch_geometric.loader import ClusterData, ClusterLoader
from multi_trail_gnas.multi_trail_evaluation import MultiTrailEvaluation
from multi_trail_gnas.discrete_search_space import search_space, gnn_topology, component_candidate_dict

class AutoGNAS(object):

    def __init__(self,
                 estimator,
                 search_parameter):

        self.estimator = estimator
        self.sharing_population_size = search_parameter["sharing_population_size"]
        self.parent_num = search_parameter["parent_num"]
        self.mutation_num = search_parameter["mutation_num"]
        self.history_gnn = []
        self.history_gnn_val_score = []

    def search(self, num_population, search_epoch, return_top_k):

        print("Multi-trail AutoGNAS Search Strategy Starting")
        print(64 * "=")

        # population initialization
        search_space_size = len(search_space)
        uniform_random_sample_population_index = random.sample(range(0, search_space_size - 1), num_population)
        population = [search_space[index] for index in uniform_random_sample_population_index]

        print("Uniform Random Initialization Population Size is:", len(population))

        # population gnn evaluate
        population_val_score = []
        for gnn in population:
            val_score = self.estimator.get_estimation_score(architecture=gnn)
            population_val_score.append(val_score)

        print(len(population), "Individuals Have Been Estimated")

        # population gnn and val score add in history gnn
        self.history_gnn = self.history_gnn + population
        self.history_gnn_val_score = self.history_gnn_val_score + population_val_score

        sharing_population, sharing_performance = self.top_population_select(population,
                                                                             population_val_score,
                                                                             top_k=self.sharing_population_size)

        # mutation select probability vector calculate
        sharing_population_temp = sharing_population.copy()
        mutation_selection_probability = self.mutation_selection_probability(sharing_population_temp,
                                                                             gnn_topology)

        print(35 * "=", "Initialization Mutation Select Probability Vector:", 35 * "=")
        print(mutation_selection_probability)
        print(35 * "=", "Constrained Evolution Start", 35 * "=")

        for epoch in range(int(search_epoch)):

            # produce new sharing population based on constrained evolution
            sharing_population, sharing_performance = self.constrained_evolution(sharing_population=sharing_population,
                                                                                 sharing_performance=sharing_performance,
                                                                                 mutation_selection_probability=mutation_selection_probability)

            # mutation select probability vector recalculate based on new sharing population
            sharing_population_temp = copy.deepcopy(sharing_population)
            mutation_selection_probability = self.mutation_selection_probability(sharing_population_temp,
                                                                                 gnn_topology)

            print("Epoch", epoch+1)
            print("Mutation Select Probability Vector:", mutation_selection_probability)
            print("\n")

        # select top gnn based on sharing population
        top_gnn, top_gnn_performance = self.top_population_select(sharing_population,
                                                                  sharing_performance,
                                                                  top_k=return_top_k)

        print("Sampled Top", return_top_k, "GNN Architecture and Estimation Score (Validation Accuracy):")
        for gnn, val_score in zip(top_gnn, top_gnn_performance):
            print(gnn, val_score)

        print(64 * "=")
        print("Multi-trail AutoGNAS Search Strategy Completion")
        return top_gnn, top_gnn_performance

    def constrained_evolution(self,
                              sharing_population,
                              sharing_performance,
                              mutation_selection_probability):

        print(35 * "=", "Constrained Evolution Search", 35 * "=")
        print("Sharing Population:")
        for gnn in sharing_population:
            print(gnn)
        print("Sharing Performance:\n", sharing_performance)
        print("[Sharing Population Performance] Mean/Median/Best:\n",
              np.mean(sharing_performance),
              np.median(sharing_performance),
              np.max(sharing_performance))

        # select a parent based on wheel strategy.
        parent = self.selection(sharing_population, sharing_performance)
        print("Parent:")
        for gnn in parent:
            print(gnn)

        # mutation based on mutation_select_probability
        children = self.mutation(parent, mutation_selection_probability,)
        print("Children:")
        for gnn in children:
            print(gnn)

        # estimate children
        children_performance = []
        for gnn in children:
            val_score = self.estimator.get_estimation_score(architecture=gnn)
            children_performance.append(val_score)

        # update sharing population based on children
        sharing_population, sharing_performance = self.update(children, children_performance,
                                                              sharing_population, sharing_performance)
        return sharing_population, sharing_performance

    def selection(self,
                  population,
                  performance):

        print(35 * "=", "Select Parents Based On Wheel Strategy", 35 * "=")
        fitness = np.array(performance)
        fitness_probability = fitness / sum(fitness)
        fitness_probability = fitness_probability.tolist()
        index_list = [index for index in range(len(fitness))]
        parent = []
        parent_index = np.random.choice(index_list, self.parent_num, replace=False, p=fitness_probability)

        for index in parent_index:
            parent.append(population[index].copy())

        return parent

    def mutation(self,
                 parent,
                 mutation_selection_probability):

        print(35 * "=", "Mutation Based On Mutation Select Probability", 35 * "=")
        for index in range(len(parent)):

            # stopping until sampling the new gnn architecture which not in the history gnn
            while parent[index] in self.history_gnn:

                # confirm mutation point in the gnn architecture genetic list based on information entropy probability
                position_to_mutate_list = np.random.choice([gene for gene in range(len(parent[index]))],
                                                           self.mutation_num,
                                                           replace=False,
                                                           p=mutation_selection_probability)

                for mutation_index in position_to_mutate_list:
                    mutation_space = component_candidate_dict[gnn_topology[mutation_index]]
                    new_candidate = np.random.choice(mutation_space, 1)[0]
                    parent[index][mutation_index] = new_candidate

        children = parent
        self.history_gnn = self.history_gnn + children
        return children

    def update(self,
               children,
               children_performance,
               sharing_population,
               sharing_performance):

        print(35 * "=", "Updating", 35 * "=")
        print("Before Sharing Population:")
        for gnn in sharing_population:
            print(gnn)
        print("Before Sharing Performance:\n", sharing_performance)

        # calculating the threshold based on sharing population
        _, top_performance = self.top_population_select(sharing_population,
                                                        sharing_performance,
                                                        top_k=self.sharing_population_size)
        threshold = np.mean(top_performance)

        index = 0
        for performance in children_performance:
            if performance > threshold:
                sharing_performance.append(performance)
                sharing_population.append(children[index])
                index += 1
            else:
                index += 1

        print("After Sharing Population:")
        for gnn in sharing_population:
            print(gnn)
        print("After Sharing Performance:\n", sharing_performance)

        return sharing_population, sharing_performance

    def top_population_select(self,
                              population,
                              performance,
                              top_k):

        population_dict = {}
        for key, value in zip(population, performance):
            population_dict[str(key)] = value

        rank_population_dict = sorted(population_dict.items(), key=lambda x: x[1], reverse=True)
        sharing_population = []
        sharing_validation_performance = []

        i = 0
        for key, value in rank_population_dict:

            if i == top_k:
                break
            else:
                sharing_population.append(eval(key))
                sharing_validation_performance.append(value)
                i += 1

        return sharing_population, sharing_validation_performance

    def information_entropy(self, p_list):

        statistical_dict = {}
        length = len(p_list)
        for key in p_list:
            statistical_dict[key] = statistical_dict.get(key, 0) + 1

        p_list = []
        for key in statistical_dict:
            p_list.append(statistical_dict[key] / length)

        p_array = np.array(p_list)
        log_p = np.log2(p_array)
        information_entropy = -sum(p_array * log_p)

        return information_entropy

    def mutation_selection_probability(self,
                                       sharing_population,
                                       gnn_architecture_flow):

        p_list = []
        for i in range(len(gnn_architecture_flow)):
            p_list.append([])

        while sharing_population:
            gnn = sharing_population.pop()
            for index in range(len(p_list)):
                p_list[index].append(gnn[index])

        gene_information_entropy = []
        for sub_list in p_list:
            gene_information_entropy.append(self.information_entropy(sub_list))

        exp_x = np.exp(gene_information_entropy)
        probability = exp_x / np.sum(exp_x)

        return probability

if __name__ == "__main__":

    data_name = "CS"
    graph = GraphData(data_name, shuffle=False).data
    cluster_data = ClusterData(graph, num_parts=1)
    graph_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gnn_model_config = {"num_node_features": graph.num_node_features,
                        "num_classes": graph.num_classes,
                        "hidden_dimension": 128,
                        "node_element_dropout_probability": 0.5,
                        "edge_dropout_probability": 0.3,
                        "learn_rate": 0.0001,
                        "weight_decay": 0.0001,
                        "train_epoch": 10}

    estimator = MultiTrailEvaluation(gnn_model_config=gnn_model_config,
                                     graph=graph_loader,
                                     device=device)

    search_parameter = {"sharing_population_size": 4,
                        "parent_num": 2,
                        "mutation_num": 2}

    searcher = AutoGNAS(estimator=estimator,
                        search_parameter=search_parameter)

    top_gnn, top_gnn_score = searcher.search(num_population=5, search_epoch=10, return_top_k=2)