import torch
import random
import numpy as np
from torch_geometric.datasets.coauthor import Coauthor
from torch_geometric.datasets.amazon import Amazon
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset
import os

class GraphData(object):

    def __init__(self, data_name, shuffle=False):

        if data_name in ["CS", "Physics", "Photo", "Computers", "ogbn-arxiv",
                         "ogbn-products", "Cora", "CiteSeer", "Pubmed"]:

            data_path = os.path.abspath(__file__)[:-13] + "/dataset"
            train_index = None
            val_index = None
            test_index = None

            if data_name in ["CS", "Physics", "Computers", "Photo", "Cora", "CiteSeer", "Pubmed"]:

                if data_name in ["CS", "Physics"]:
                    data = Coauthor(data_path, data_name)
                elif data_name in ["Computers", "Photo"]:
                    data = Amazon(data_path, data_name)
                elif data_name in ["Cora", "CiteSeer", "Pubmed"]:
                    data = Planetoid(data_path, data_name)
                else:
                    raise Exception("Sorry current version don't "
                                    "Support this default datasets", data_name)

                self.data = data[0]
                self.data.num_classes = data.num_classes

                train_ratio = None
                val_ratio = None
                test_ratio = None
                if data_name == "CS":
                    train_ratio = 3000
                    val_ratio = 450
                elif data_name == "Physics":
                    train_ratio = 500
                    val_ratio = 150
                elif data_name == "Computers":
                    train_ratio = 200
                    val_ratio = 300
                elif data_name == "Photo":
                    train_ratio = 3500
                    val_ratio = 240
                elif data_name == "Cora" or data_name == "CiteSeer" or data_name == "Pubmed":
                    # train_ratio = self.count_(self.data.train_mask)
                    train_ratio = 80
                    val_ratio = self.count_(self.data.val_mask)
                    test_ratio = self.count_(self.data.test_mask)

                # train / val / test mask construction
                index = [i for i in range(self.data.num_nodes)]

                if shuffle:
                    random.shuffle(index)

                train_index = index[:train_ratio]
                val_index = index[train_ratio:train_ratio+val_ratio]

                if data_name in ["CS", "Photo", "Computers", "Physics"]:
                    test_index = index[train_ratio+val_ratio:]
                else:
                    test_index = index[train_ratio+val_ratio:train_ratio+val_ratio+test_ratio]

            elif data_name in ["ogbn-arxiv", "ogbn-products"]:

                data = PygNodePropPredDataset(name=data_name, root=data_path)
                split_idx = data.get_idx_split()

                train_index = split_idx["train"].numpy().tolist()
                val_index = split_idx["valid"].numpy().tolist()
                test_index = split_idx["test"].numpy().tolist()

                self.data = data[0]
                self.data.y = self.data.y.squeeze(dim=1)

                self.data.num_classes = data.num_classes
                self.data.train_idx = train_index
                self.data.val_idx = val_index
                self.data.test_idx = test_index


            train_mask = torch.tensor(self.mask(train_index, self.data.num_nodes), dtype=torch.bool)
            val_mask = torch.tensor(self.mask(val_index, self.data.num_nodes), dtype=torch.bool)
            test_mask = torch.tensor(self.mask(test_index, self.data.num_nodes), dtype=torch.bool)

            self.data.train_index = train_index
            self.data.val_index = val_index
            self.data.test_index = test_index

            self.data.data_name = data_name
            self.data.train_mask = train_mask
            self.data.val_mask = val_mask
            self.data.test_mask = test_mask

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.data.to(device)
        else:
            raise Exception("Sorry current version don't "
                            "Support this default datasets", data_name)

    def mask(self, index, num_node):
        """ create mask """
        mask = np.zeros(num_node)
        for idx in index:
            mask[idx] = 1
        return np.array(mask, dtype=np.int32)

    def count_(self, mask):
        true_num = 0
        for i in mask:
            if i:
                true_num += 1
        return true_num

if __name__=="__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data_name = "Cora"
    # data_name = "CiteSeer"
    # data_name = "Pubmed"
    # data_name = "Photo"
    # data_name = "ogbn-products"
    # data_name = "Physics"
    data_name = "CS"
    # data_name = "Computers"
    graph = GraphData(data_name, shuffle=False).data
    print("Data set", data_name, "Loading Success")
