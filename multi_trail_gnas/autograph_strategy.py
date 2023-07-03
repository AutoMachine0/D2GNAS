import copy
import torch
import random
from planetoid import GraphData
from multi_trail_gnas.multi_trail_evaluation import MultiTrailEvaluation
from torch_geometric.loader import ClusterData, ClusterLoader
from multi_trail_gnas.discrete_search_space import search_space, conv_candidate, norm_candidate, act_candidate

class AutoGraph(object):

    def __init__(self, estimator):

        self.estimator = estimator
        self.history_gnn = []
        self.history_gnn_val_score = []

    def search(self, num_population, search_epoch, return_top_k):
        print("Multi-trail AutoGraph Search Strategy Starting")
        print(64*"=")

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

        # regularized evolution search start
        for epoch in range(search_epoch):
            print(32*"+")
            print("Search Epoch:", epoch+1)

            best_val_score_index = population_val_score.index(max(population_val_score))
            best_individual = copy.deepcopy(population[best_val_score_index])
            best_individual_val_score = population_val_score[best_val_score_index]

            print(" Best Individual:", best_individual, "Validation Score:", best_individual_val_score)

            population, population_val_score = self.age_mutation(population,
                                                                 population_val_score,
                                                                 best_individual)
        # select top n gnn in history gnn
        top_gnn, top_gnn_val_score = self.estimator.rank_based_estimation_score(self.history_gnn,
                                                                                self.history_gnn_val_score,
                                                                                top_k=return_top_k)

        print("Sampled Top", return_top_k, "GNN Architecture and Estimation Score (Validation Accuracy):")
        for gnn, val_score in zip(top_gnn, top_gnn_val_score):
            print(gnn, val_score)

        print(64 * "=")
        print("Multi-trail AutoGraph Search Strategy Completion")
        return top_gnn, top_gnn_val_score

    def age_mutation(self, population, population_val_score, best_individual):

        # random single point mutation
        individual_mutation_index = random.randint(0, len(best_individual)-1)
        print("Mutation Index:", individual_mutation_index)
        mutation_operator = None
        if individual_mutation_index in [0, 3]:
            # conv mutation
            print("Convolution Operation Mutation")

            mutation_candidate_index = random.randint(0, len(conv_candidate)-1)
            # generate new gene
            while conv_candidate[mutation_candidate_index] == best_individual[individual_mutation_index]:
                mutation_candidate_index = random.randint(0, len(conv_candidate) - 1)

            mutation_operator = conv_candidate[mutation_candidate_index]

        elif individual_mutation_index in [1, 4]:
            # norm mutate
            print("Normalization Operation Mutation")

            mutation_candidate_index = random.randint(0, len(norm_candidate) - 1)
            # generate new gene
            while norm_candidate[mutation_candidate_index] == best_individual[individual_mutation_index]:
                mutation_candidate_index = random.randint(0, len(norm_candidate) - 1)

            mutation_operator = norm_candidate[mutation_candidate_index]

        elif individual_mutation_index in [2, 5]:
            # act mutate
            print("Activation Operation Mutation")

            mutation_candidate_index = random.randint(0, len(act_candidate) - 1)
            # generate new gene
            while act_candidate[mutation_candidate_index] == best_individual[individual_mutation_index]:
                mutation_candidate_index = random.randint(0, len(act_candidate) - 1)

            mutation_operator = act_candidate[mutation_candidate_index]

        print("Original Individual Index", individual_mutation_index,
              "Operation Mutation Situation:", best_individual[individual_mutation_index],
              "--->", mutation_operator)

        best_individual[individual_mutation_index] = mutation_operator

        # child gnn evaluate
        val_score = self.estimator.get_estimation_score(architecture=best_individual)
        print("Child Individual:", best_individual, "Estimation Score:", val_score)

        # age evolution strategy
        population.append(best_individual)  # child gnn add in the far right end of population queue
        population_val_score.append(val_score)

        population.pop(0)  # remove the child gnn from the far left end of population queue
        population_val_score.pop(0)
        # add child gnn information in history list
        self.history_gnn.append(best_individual)
        self.history_gnn_val_score.append(val_score)

        return population, population_val_score

if __name__ == "__main__":

    data_name = "Cora"
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
                        "train_epoch": 50}

    estimator = MultiTrailEvaluation(gnn_model_config=gnn_model_config,
                                     graph=graph_loader,
                                     device=device)

    searcher = AutoGraph(estimator=estimator)

    top_gnn, top_gnn_score = searcher.search(num_population=10, search_epoch=10, return_top_k=2)
