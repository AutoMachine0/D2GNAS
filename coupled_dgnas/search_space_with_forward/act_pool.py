import torch.nn as nn
import torch.nn.functional

class ActPool(nn.Module):

    def __init__(self, act_name="Relu"):

        super(ActPool, self).__init__()

        self.candidate_list = ["Elu", "LeakyRelu",
                               "Relu", "Relu6",
                               "Sigmoid", "Softplus",
                               "Tanh", "Linear"]

        if act_name == "Elu":
            self.act_operation = torch.nn.functional.elu
        elif act_name == "LeakyRelu":
            self.act_operation = torch.nn.functional.leaky_relu
        elif act_name == "Relu":
            self.act_operation = torch.nn.functional.relu
        elif act_name == "Relu6":
            self.act_operation = torch.nn.functional.relu6
        elif act_name == "Sigmoid":
            self.act_operation = torch.sigmoid
        elif act_name == "Softplus":
            self.act_operation = torch.nn.functional.softplus
        elif act_name == "Tanh":
            self.act_operation = torch.tanh
        elif act_name == "Linear":
            self.act_operation = lambda x: x
        else:
            raise Exception("Sorry current version don't "
                            "Support this default act", act_name)

    def forward(self, x):

        return self.act_operation(x)

if __name__=="__main__":
   a = ActPool("Elu")
   b = 1