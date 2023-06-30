import torch.nn.functional as F
from torch.nn import Linear
import torch

class MessageNoneAgg(torch.nn.Module):

    def __init__(self):
        super(MessageNoneAgg, self).__init__()


    def forward(self, M):

        if len(list(M)) > 3:
            raise ("M dimension wrong:", len(list(M)), "It should be 2 for MessageNoneAgg")

        c = M

        return c


class MessageSumAgg(torch.nn.Module):

    def __init__(self):
        super(MessageSumAgg, self).__init__()


    def forward(self, M):

        c = torch.sum(M, 0)

        return c


class MessageMeanAgg(torch.nn.Module):

    def __init__(self):
        super(MessageMeanAgg, self).__init__()

    def forward(self, M):

        c = torch.sum(M, 0) / M.size()[0]

        return c


class MessageMaxAgg(torch.nn.Module):

    def __init__(self):
        super(MessageMaxAgg, self).__init__()


    def forward(self, M):

        c = torch.max(M, 0)[0]

        return c


class MessageCatAgg(torch.nn.Module):

    def __init__(self):
        super(MessageCatAgg, self).__init__()

    def forward(self, M):

        c = None

        for m_index in range(M.size()[0]):
            if not m_index == 0:
                c = M[m_index]
            else:
                c = torch.cat((c, M[m_index]), dim=1)

        return c


class MessageWeightAgg(torch.nn.Module):

    def __init__(self):
        super(MessageWeightAgg, self).__init__()

    def foward(self, M):
        return M


class MessageAdaAgg(torch.nn.Module):

    def __init__(self, feat_dim):
        super(MessageAdaAgg, self).__init__()
        self.learnable_weight_layer = Linear(feat_dim, 1)

    def forward(self, M):

        weight_matrix = F.softmax(torch.sigmoid(self.learnable_weight_layer(M)), dim=0)
        M_weight = M * weight_matrix
        c = torch.sum(M_weight, 0)

        return c

if __name__=="__main__":
    pass