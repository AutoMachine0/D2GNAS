from hyperopt import hp

HP_SEARCH_SPACE = {"weight_decay": hp.choice("weight_decay", [0, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
                   "learning_rate": hp.choice("learning_rate", [1e-1, 1e-2, 5e-3, 1e-3, 5e-4]),
                   "node_element_dropout_probability": hp.choice("node_element_dropout_probability", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                   "edge_dropout_probability": hp.choice("edge_dropout_probability", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                   "hidden_dimension": hp.choice("hidden_dimension", [8, 16, 32, 64, 128, 256, 512, 1024])}


HP_SEARCH_SPACE_Mapping = {"weight_decay": [0, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
                           "learning_rate": [1e-1, 1e-2, 5e-3, 1e-3, 5e-4],
                           "node_element_dropout_probability": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                           "edge_dropout_probability": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                           "hidden_dimension": [8, 16, 32, 64, 128, 256, 512, 1024]}