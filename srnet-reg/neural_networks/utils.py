import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float64) # 接受数据和标签作为输入，并将它们转换为PyTorch张量。这里使用torch.tensor将数据转换为浮点数张量。
        self.labels = labels
        assert self.data.shape[0] == self.labels.shape[0] # 使用assert语句来确保数据的数量与标签的数量一致。

    # 返回数据集中的样本数量，即数据的行数。
    def __len__(self):
        return self.data.shape[0]
    
    # 通过索引来访问数据集中的单个样本。它返回给定索引的输入数据和相应的标签。
    def __getitem__(self, index):
        return self.data[index, :], self.labels[index]
    
# 总的来说，这个类是一个自定义的PyTorch数据集，它能够将提供的数据和标签封装起来，并提供标准的接口来访问这些数据。这为使用PyTorch的数据加载器和其他工具提供了便利。