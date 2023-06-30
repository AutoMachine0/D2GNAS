from scipy import sparse
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pre_processing(x, A, k, graph_aggregator, alpha=0):


    if (not isinstance(A, np.ndarray)) or (not isinstance(x, np.ndarray)):
        raise ("x and A need be ndarray type")

    A = sparse.csr_matrix(A)
    spread_matrix = graph_aggregator()
    S = spread_matrix.get(A)
    M = [x]

    for _ in range(k):

        if spread_matrix.name == "PerPage":
            m_k = alpha * x + (1 - alpha) * S.dot(M[-1])
        else:
            m_k = S.dot(M[-1])

        M.append(m_k)

    M = torch.stack([torch.FloatTensor(m_k).to(device) for m_k in M[1:]], dim=0)

    return M

if __name__ == "__main__":

    from autognas.datasets.util.util_cite_t import DATA
    from graph_agg import Augmented_Normalized_Adjacency
    import time
    graph = DATA()
    dataset = "citeseer"

    x, _, A = graph.get_data(dataset)
    x = x.numpy()
    A = A.numpy()
    t = time.time()
    M = pre_processing(x=x,
                       A=A,
                       k=6,
                       graph_aggregator=Augmented_Normalized_Adjacency,
                       alpha=0.3)

    time_cost = time.time() - t
    print("cost time", time_cost)