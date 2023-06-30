from scipy import sparse
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def post_processing(x, A, k, graph_aggregator, alpha):

    if (not torch.is_tensor(x)) or (not isinstance(A, np.ndarray)):
        raise TypeError("The x must be tensor, and A must be numpy ndarray")

    if x.device == "cpu":
        x = x.detach().numpy()
    else:
        x = x.to("cpu").detach().numpy()

    A = sparse.csr_matrix(A)
    spread_matrix = graph_aggregator()
    S = spread_matrix.get(A)

    for _ in range(k):

        if spread_matrix.name == "PerPage":
            x = alpha * x + (1 - alpha) * S.dot(x)
        else:
            x = S.dot(x)

    return torch.tensor(x).to(device)

if __name__=="__main__":
    from autognas.datasets.util.util_cite_t import DATA
    from graph_agg import Personalized_Pagerank
    import time

    graph = DATA()
    dataset = "cora"

    x, _, A = graph.get_data(dataset)
    x = x.to("cuda")
    A = A.numpy()
    t = time.time()
    M = post_processing(x=x,
                        A=A,
                        k=6,
                        graph_aggregator=Personalized_Pagerank,
                        alpha=0.3)
    time_cost = time.time() - t
    print("cost time", time_cost)