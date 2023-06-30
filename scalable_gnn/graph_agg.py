from scipy import sparse
import numpy as np
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Augmented_Normalized_Adjacency():

    def __init__(self):
        self.name = "AugNorAdj"

    def get(self, A):
        """
           A' = A + I
           D' = D + I
           S = D'(-1/2)A'D'(-1/2)
        """

        A = A + sparse.eye(A.shape[0])
        D = np.array(A.sum(1))
        D_inv_sqrt = np.power(D, -0.5).flatten()
        # fix inf in the D_inv_sqrt ndarray
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
        D_inv_sqrt = sparse.diags(D_inv_sqrt)
        S = D_inv_sqrt.dot(A).dot(D_inv_sqrt)

        return S


class Augmented_Random_Walk():

    def __init__(self):
        self.name = "AugRanWalk"

    def get(self, A):
        """
           A' = A + I
           D' = D + I
           S = D'(-1/2)A'D'(-1/2)
           """

        A = A + sparse.eye(A.shape[0])
        D = np.array(A.sum(1))
        D_inv_sqrt = np.power(D, -1).flatten()
        # fix inf in the D_inv_sqrt ndarray
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
        D_inv_sqrt = sparse.diags(D_inv_sqrt)
        S = D_inv_sqrt.dot(A)

        return S


class Personalized_Pagerank():

    def __init__(self):
        self.name = "PerPage"

    def get(self, A):
        """
           A' = A + I
           D' = D + I
           S = D'(-1/2)A'D'(-1/2)
           """

        A = A + sparse.eye(A.shape[0])
        D = np.array(A.sum(1))
        D_inv_sqrt = np.power(D, -0.5).flatten()
        # fix inf in the D_inv_sqrt ndarray
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
        D_inv_sqrt = sparse.diags(D_inv_sqrt)
        S = D_inv_sqrt.dot(A).dot(D_inv_sqrt)

        return S


class First_Order_Cheby():

    def __init__(self):
        self.name = "FirOrdChe"

    def get(self, A):
        """
        S = I + D(-1/2)AD(-1/2)
        """
        I = sparse.eye(A.shape[0])
        A = A
        D = np.array(A.sum(1))
        D_inv_sqrt = np.power(D, -0.5).flatten()
        # fix inf in the D_inv_sqrt ndarray
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
        D_inv_sqrt = sparse.diags(D_inv_sqrt)
        S = I + D_inv_sqrt.dot(A).dot(D_inv_sqrt)

        return S

if __name__ == "__main__":
    pass