from scalable_gnn.base_op import ActFunction
from coupled_dgnas.search_space_with_forward.norm_pool import NormPool
import torch

class MLPUpdator(torch.nn.Module):

    def __init__(self, input_dim, output_dim, dropout_rate, hidden_dim_list, act_list, norm_name):

        if not isinstance(hidden_dim_list, list) and isinstance(act_list, list):
            raise TypeError("The x and A must be list")

        super(MLPUpdator, self).__init__()

        self.mlp = torch.nn.Sequential()

        self.activator = ActFunction()
        self.dropout = torch.nn.Dropout(dropout_rate)

        # single layer
        if len(hidden_dim_list) == 0:

            self.mlp.add_module("layer 1 dropout", self.dropout)
            self.mlp.add_module("layer 1 weight", torch.nn.Linear(input_dim, output_dim))

        if len(hidden_dim_list) == 1:

            self.mlp.add_module("layer 1 dropout", self.dropout)
            self.mlp.add_module("layer 1 weight", torch.nn.Linear(input_dim, hidden_dim_list[0]))
            self.mlp.add_module("layer 1 norm", NormPool(input_dim=input_dim, norm_name=norm_name))
            self.mlp.add_module("layer 1 activation", self.activator.activation_get(act_list[0]))

            self.mlp.add_module("layer 2 dropout", self.dropout)
            self.mlp.add_module("layer 2 weight", torch.nn.Linear(hidden_dim_list[0], output_dim))

        # multiple layers
        if len(hidden_dim_list) > 1:

            for i in range(len(hidden_dim_list)):
                # input layer
                if i == 0:
                    self.mlp.add_module("layer 1 dropout", self.dropout)
                    self.mlp.add_module("layer 1 weight", torch.nn.Linear(input_dim, hidden_dim_list[i]))
                    self.mlp.add_module("layer 1 norm", NormPool(input_dim=hidden_dim_list[i], norm_name=norm_name))
                    self.mlp.add_module("layer 1 activation", self.activator.activation_get(act_list[i]))
                # intermediate layer
                else:
                    self.mlp.add_module("layer " + str(i + 1) + " dropout", self.dropout)
                    self.mlp.add_module("layer " + str(i + 1) + " weight", torch.nn.Linear(hidden_dim_list[i-1], hidden_dim_list[i]))
                    self.mlp.add_module("layer " + str(i + 1) + " norm", NormPool(input_dim=hidden_dim_list[i], norm_name=norm_name))
                    self.mlp.add_module("layer " + str(i + 1) + " activation", self.activator.activation_get(act_list[i]))

            # output layer
            self.mlp.add_module("layer " + str(len(hidden_dim_list) + 1) + " dropout", self.dropout)
            self.mlp.add_module("layer " + str(len(hidden_dim_list) + 1) + " weight",
                                torch.nn.Linear(hidden_dim_list[-1], output_dim))


    def forward(self, x):

        output = self.mlp(x)

        return output

if __name__=="__main__":

    mymodel = MLPUpdator(input_dim=100,
                         output_dim=10,
                         dropout_rate=0.8,
                         hidden_dim_list=[64],
                         act_list=["relu"],
                         norm_name="GraphNorm")
    pass