import torch

class MLP(torch.nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 dropout_rate=0.0,
                 hidden_dim_list=[],
                 act_list=[]):
        """
        hidden_dim_list元素个数必须与act_list元素个数相同
        """

        if not isinstance(hidden_dim_list, list) and isinstance(act_list, list):
            raise TypeError("The x and A must be list")

        super(MLP, self).__init__()
        self.mlp = torch.nn.Sequential()
        self.activator = ActPool()

        self.dropout = torch.nn.Dropout(dropout_rate)

        # single layer
        if len(hidden_dim_list) == 0:

            self.mlp.add_module("layer 1 dropout", self.dropout)
            self.mlp.add_module("layer 1 weight", torch.nn.Linear(input_dim, output_dim))

        if len(hidden_dim_list) == 1:

            self.mlp.add_module("layer 1 dropout", self.dropout)
            self.mlp.add_module("layer 1 weight", torch.nn.Linear(input_dim, hidden_dim_list[0]))
            self.mlp.add_module("layer 1 activation", self.activator.get_act(act_list[0]))

            self.mlp.add_module("layer 2 dropout", self.dropout)
            self.mlp.add_module("layer 2 weight", torch.nn.Linear(hidden_dim_list[0], output_dim))

        # multiple layers
        if len(hidden_dim_list) > 1:

            for i in range(len(hidden_dim_list)):
                # input layer
                if i == 0:
                    self.mlp.add_module("layer 1 dropout", self.dropout)
                    self.mlp.add_module("layer 1 weight", torch.nn.Linear(input_dim, hidden_dim_list[i]))
                    self.mlp.add_module("layer 1 activation", self.activator.get_act(act_list[i]))
                # intermediate layer
                else:
                    self.mlp.add_module("layer " + str(i + 1) + " dropout", self.dropout)
                    self.mlp.add_module("layer " + str(i + 1) + " weight", torch.nn.Linear(hidden_dim_list[i-1], hidden_dim_list[i]))
                    self.mlp.add_module("layer " + str(i + 1) + " activation", self.activator.get_act(act_list[i]))

            # output layer
            self.mlp.add_module("layer " + str(len(hidden_dim_list) + 1) + " dropout", self.dropout)
            self.mlp.add_module("layer " + str(len(hidden_dim_list) + 1) + " weight",
                                torch.nn.Linear(hidden_dim_list[-1], output_dim))


    def forward(self, x):

        output = self.mlp(x)

        return output

class ActPool(object):

    def __init__(self):
        pass

    def get_act(self, act_name):

        if act_name == "Elu":
            act = torch.nn.ELU()
        elif act_name == "LeakyRelu":
            act = torch.nn.LeakyReLU()
        elif act_name == "Relu":
            act = torch.nn.ReLU()
        elif act_name == "Relu6":
            act = torch.nn.ReLU6()
        elif act_name == "Sigmoid":
            act = torch.nn.Sigmoid()
        elif act_name == "Softplus":
            act = torch.nn.Softplus()
        elif act_name == "Tanh":
            act = torch.nn.Tanh()
        elif act_name == "Linear":
            act = Linear()
        else:
            raise Exception("Sorry current version don't "
                            "Support this default act", act_name)

        return act

class Linear(torch.nn.Module):

    def __init__(self):
        super(Linear, self).__init__()

    def forward(self, x):
        return x


if __name__=="__main__":

    # model = MLP(input_dim=100,
    #             output_dim=10,
    #             dropout_rate=0.8,
    #             hidden_dim_list=[64, 64],
    #             act_list=["Relu", "Relu"])
    #
    # model = MLP(input_dim=100,
    #             output_dim=10,
    #             dropout_rate=0.0,
    #             hidden_dim_list=[64, 64],
    #             act_list=["Linear", "Linear"])

    model = MLP(input_dim=100,
                output_dim=100)

    x = torch.ones(100)
    linear = Linear()
    x_ = linear(x)
    y = model(x)

    pass