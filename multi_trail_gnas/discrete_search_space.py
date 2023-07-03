import os
from search_space.act_pool import ActPool
from search_space.conv_pool import ConvPool
from search_space.norm_pool import NormPool

gnn_topology = ["Convolution", "Normalization", "Activation",
                "Convolution", "Normalization", "Activation"]

conv_candidate = ConvPool().candidate_list
norm_candidate = NormPool().candidate_list
act_candidate = ActPool().candidate_list

component_candidate_dict = {"Convolution": conv_candidate,
                            "Normalization": norm_candidate,
                            "Activation": act_candidate}

def search_space_candidate(gnn_candidates_path):
    print("Obtaining Discrete Search Space")
    gnn_architecture_candidates = []

    if os.path.exists(gnn_candidates_path):
        print("Read Discrete Search Space From:", gnn_candidates_path)
        with open(gnn_candidates_path, "r") as f:
            gnn_architecture_candidates = f.readlines()
            gnn_architecture_candidates = [gnn.replace("\n", "").split(" ") for gnn in gnn_architecture_candidates]
    else:
        print("Create Discrete Search Space")
        for layer_1_conv in conv_candidate:
            for layer_1_norm in norm_candidate:
                for layer_1_act in act_candidate:
                    for layer_2_conv in conv_candidate:
                        for layer_2_norm in norm_candidate:
                            for layer_2_act in act_candidate:
                                gnn_archi = layer_1_conv + " " + layer_1_norm + " " + layer_1_act + \
                                            " " + layer_2_conv + " " + layer_2_norm + " " + layer_2_act
                                gnn_architecture_candidates.append(gnn_archi)
                                with open(gnn_candidates_path, "a+") as f:
                                    f.write(gnn_archi + "\n")
        print("Discrete Search Space Create Completion")
    print("Discrete Search Space Obtained")
    return gnn_architecture_candidates

search_space_path = os.path.abspath(__file__)[:-41] + "search_space_gnn_candidates.txt"

search_space = search_space_candidate(search_space_path)

if __name__=="__main__":
    pass
