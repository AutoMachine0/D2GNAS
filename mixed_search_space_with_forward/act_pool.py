import torch.nn as nn
import torch.nn.functional

class ActPool(nn.Module):

    def __init__(self):

        super(ActPool, self).__init__()

        self.candidate_list = ["Elu", "LeakyRelu",
                               "Relu", "Relu6",
                               "Sigmoid", "Softplus",
                               "Tanh", "Linear"]

    def get_candidate(self, candidate_name):

        if candidate_name == "Elu":
            act_operation = torch.nn.functional.elu
        elif candidate_name == "LeakyRelu":
            act_operation = torch.nn.functional.leaky_relu
        elif candidate_name == "Relu":
            act_operation = torch.nn.functional.relu
        elif candidate_name == "Relu6":
            act_operation = torch.nn.functional.relu6
        elif candidate_name == "Sigmoid":
            act_operation = torch.sigmoid
        elif candidate_name == "Softplus":
            act_operation = torch.nn.functional.softplus
        elif candidate_name == "Tanh":
            act_operation = torch.tanh
        elif candidate_name == "Linear":
            act_operation = lambda x: x
        else:
            raise Exception("Sorry current version don't "
                            "Support this default act", candidate_name)
        return act_operation

    def forward(self, x):

        return self.act_operation(x)

if __name__=="__main__":
   a = ActPool()
   b = 1