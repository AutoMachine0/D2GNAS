import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActFunction():

    def __init__(self):
        pass

    def activation_get(self, act):


        if act == "elu":
            act = torch.nn.ELU()
        elif act == "leaky_relu":
            act = torch.nn.LeakyReLU()
        elif act == "relu":
            act = torch.nn.ReLU()
        elif act == "relu6":
            act = torch.nn.ReLU6()
        elif act == "sigmoid":
            act = torch.nn.Sigmoid()
        elif act == "softplus":
            act = torch.nn.Softplus()
        elif act == "tanh":
            act = torch.nn.Tanh()
        else:
            raise ("wrong act type:", act)

        return act


class LossFunction():

    def __init__(self):
        pass

    def loss_get(self, loss):

        loss_f = None
        # multi classification loss
        if loss == "crossentropy":
            loss_f = torch.nn.CrossEntropyLoss()
        elif loss == "nllloss":
            loss_f = torch.nn.NLLLoss()
        elif loss == "multilabelloss":
            loss_f = torch.nn.MultiLabelSoftMarginLoss()
        elif loss == "bceloss":
            loss_f = torch.nn.BCEWithLogitsLoss()

        return loss_f


class OptimationFunction():

    def __init__(self):
        pass

    def optimation_get(self, opt):

        optim_f = None

        if opt == "sgd":
            optim_f = torch.optim.SGD
        elif opt == "asgd":
            optim_f = torch.optim.ASGD
        elif opt == "adagrad":
            optim_f = torch.optim.Adagrad
        elif opt == "adadelta":
            optim_f = torch.optim.Adadelta
        elif opt == "rmsprop":
            optim_f = torch.optim.RMSprop
        elif opt == "adam":
            optim_f = torch.optim.Adam
        elif opt == "adamax":
            optim_f = torch.optim.Adamax
        elif opt == "sparseadam":
            optim_f = torch.optim.SparseAdam
        elif opt == "lbfgs":
            optim_f = torch.optim.LBFGS

        return optim_f

def x_row_normalization(x):
    I = torch.eye(x.shape[0]).to(device)
    nor_e = x.sum(1).pow(-1)
    nor_e = torch.where(torch.isinf(nor_e), torch.full_like(nor_e, 0.), nor_e)
    nor_e = nor_e * I
    nor_x = torch.mm(nor_e, x)
    return nor_x

def parameters_initialization(paramters, manner="uniform"):

    if manner == "uniform":
        paramters = torch.nn.init.uniform_(paramters)
    elif manner == "normal":
        paramters = torch.nn.init.normal_(paramters)
    else:
        raise TypeError("parameter initialization manner error")

    return paramters

def batch_operator(A, X, Y, batch_size=2):

    A_batch_size_list = []
    X_batch_size_list = []
    Y_batch_size_list = []
    start_index = 0

    if batch_size > 1:
        split_offset = int(len(A) / batch_size)

        while start_index < len(A):

            A_batch_size_list.append(A[start_index:start_index+split_offset])
            X_batch_size_list.append(X[start_index:start_index+split_offset])
            Y_batch_size_list.append(Y[start_index:start_index+split_offset])
            start_index = start_index + split_offset

    elif batch_size == 1:
        A_batch_size_list.append(A)
        X_batch_size_list.append(X)
        Y_batch_size_list.append(Y)

    else:
        raise TypeError("batch_size error, batch_size need greater than or equal to 1")

    A_batch_list = []
    X_batch_list = []
    Y_batch_list = []
    batch_index_list = []

    for batch_A, batch_X, batch_Y in zip(A_batch_size_list,
                                         X_batch_size_list,
                                         Y_batch_size_list):

        dim = 0
        index_list = []
        index = 0
        for a in batch_A:
            temp_index_list = []
            temp_index_list.append(index)
            index_list = index_list + temp_index_list * a.shape[0]
            dim = dim + a.shape[0]
            index += 1

        batch_index_list.append(index_list)
        merge_batch_A = np.zeros((dim, dim), dtype=np.float64)
        merge_batch_X = np.zeros((dim, batch_X[0].shape[1]))

        # for multi-label
        if batch_Y[0].ndim > 1:
            merge_batch_Y = np.zeros((len(batch_Y), batch_Y[0].shape[1]))
        else:
            merge_batch_Y = np.zeros((len(batch_Y)))


        start_index = 0
        start_index_ = 0
        for a, x, y in zip(batch_A, batch_X, batch_Y):
            offset = a.shape[0]
            merge_batch_A[start_index:start_index+offset, start_index:start_index+offset] = a
            merge_batch_X[start_index:start_index+offset, :] = x

            if batch_Y[0].ndim > 1: # for multi-label
                merge_batch_Y[start_index_, :] = y
            else:
                merge_batch_Y[start_index_] = y
            start_index = start_index + offset
            start_index_ += 1

        A_batch_list.append(merge_batch_A)
        X_batch_list.append(merge_batch_X)
        Y_batch_list.append(merge_batch_Y)

    return A_batch_list, X_batch_list, Y_batch_list, batch_index_list

if __name__=="__main__":
    A = torch.nn.Parameter(torch.Tensor(5, 3))
    print(A)
    B = parameters_initialization(A)
    print(B)
    C = parameters_initialization(A, manner='normal')
    print(C)
    pass