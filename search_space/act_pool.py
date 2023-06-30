import torch.nn.functional
from torch.nn.functional import softplus

class Linear(torch.nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, x):
        return x

class ActPool(Linear):

    def __init__(self):
        super(ActPool, self).__init__()

        self.candidate_list = ["Elu", "LeakyRelu",
                               "Relu", "Relu6",
                               "Sigmoid", "Softplus",
                               "Tanh", "Linear"]

    def get_act(self, act_name):

        if act_name == "Elu":
            act = torch.nn.functional.elu
        elif act_name == "LeakyRelu":
            act = torch.nn.functional.leaky_relu
        elif act_name == "Relu":
            act = torch.nn.functional.relu
        elif act_name == "Relu6":
            act = torch.nn.functional.relu6
        elif act_name == "Sigmoid":
            act = torch.sigmoid
        elif act_name == "Softplus":
            act = torch.nn.functional.softplus
        elif act_name == "Tanh":
            act = torch.tanh
        elif act_name == "Linear":
            # act = Linear()
            act = lambda x: x
        else:
            raise Exception("Sorry current version don't "
                            "Support this default act", act_name)
        return act

