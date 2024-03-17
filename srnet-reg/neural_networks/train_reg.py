import json
import random

import torch
from torch.autograd import Variable

from data_utils import io, draw
from dataset_config import FUNC_MAP, INTER_MAP
from neural_networks.nn_models import MLP2, MLP3

# 定义一个数据迭代器函数。它接受批量大小、输入特征x和标签y。函数会随机打乱样本索引，并按批次生成输入特征和标签。
def data_iter(batch_size, x, y):
    num_samples = x.shape[0]
    indices = list(range(num_samples))
    random.shuffle(indices)
    for i in range(0, num_samples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_samples)])
        yield x.index_select(0, j), y.index_select(0, j)

# 定义一个数据集分割函数。它根据给定的分割比例（默认为0.8）将数据集分割为训练集和测试集。
def split_dataset(dataset):
    prob = 0.8
    n_train = int(dataset.shape[0] * prob)

    return dataset[:n_train], dataset[n_train:]

# 定义一个评估损失函数。它计算给定数据迭代器和模型的平均损失。这里使用了torch.no_grad()来确保在评估过程中不计算梯度。
def evaluate_loss(data_iters, net, loss_func):
    metric_sum, n = 0.0, 0
    dev = list(net.parameters())[0].device
    with torch.no_grad():
        for x, y in data_iters:
            pred = net(x.to(dev))
            if isinstance(pred, tuple):
                pred = pred[-1]

            metric_sum += loss_func(pred, y.to(device)).cpu().item()
            n += 1

    if n == 0:
        return 0
    return metric_sum / n

# 定义一个预测函数。它接受模型和输入数据，返回模型的输出。这里同样使用了detach()来从计算图中分离张量。
def predict(net, x):
    dev = list(net.parameters())[0].device
    return list([output.cpu().detach() for output in net(x.to(dev))])


if __name__ == '__main__':

    # ediable hyparameters
    # 定义可编辑的超参数，如数据文件名、输出数量、隐藏层大小、学习率、训练周期数、批量大小、优化器和神经网络模型类型。
    data_filename = 'feynman5'
    n_output, n_hidden, lr, n_epoch, batch_size = 1, 5, 1e-3, 50000, 300
    optimizer = torch.optim.Adam
    reg_model = MLP3

    # processing data
    # 处理数据。首先加载数据集，然后将其分割为训练集和测试集。接着，分割输入特征和标签。
    dataset = io.get_dataset(f'../dataset/{data_filename}')
    data_train, data_test = split_dataset(dataset)
    n_var = data_train.shape[1] - n_output
    x_train, y_train = data_train[:, :n_var], data_train[:, n_var:]
    x_test, y_test = data_test[:, :n_var], data_test[:, n_var:]

    # train net
    # 初始化神经网络模型和优化器。模型和数据将被发送到相应的设备（CPU或CUDA）。在每个训练周期结束后，计算并记录训练集和测试集的平均损失。然后，打印出当前周期的训练和测试损失。
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mse = torch.nn.MSELoss()

    reg_model = reg_model(n_var, n_output, n_hidden).to(device)
    optimizer = optimizer(reg_model.parameters(), lr=lr)

    train_loss, test_loss = [], []
    print('start')
    for epoch in range(n_epoch):
        train_loss_sum, num_batchs = 0.0, 0
        train_iter = data_iter(batch_size, x_train, y_train)
        for x_batch, y_batch in train_iter:
            if torch.cuda.is_available():
                x_batch = Variable(x_batch).to(device)
                y_batch = Variable(y_batch).to(device)

            y_hat = reg_model(x_batch)
            if isinstance(y_hat, tuple):
                y_hat = y_hat[-1]

            loss = mse(y_hat, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.cpu().item()
            num_batchs += 1
        train_aver_loss = train_loss_sum / num_batchs
        test_aver_loss = evaluate_loss(data_iter(batch_size, x_test, y_test), reg_model, mse)

        train_loss.append(train_aver_loss)
        test_loss.append(test_aver_loss)
        print('# epoch %d: Train Loss=%.8f, Test Loss=%.8f' %
              (epoch + 1, train_aver_loss, test_aver_loss))

    # draw and save
    # 绘制训练和测试损失的趋势图，并将其保存为PDF文件。同时，创建一个目录来保存结果和相关文件。
    result_dir = f'../dataset/{data_filename}_nn/'
    io.mkdir(result_dir)

    draw.draw_mul_curves_and_save(list(range(n_epoch)), [train_loss, test_loss],
                                  savepath=result_dir+'trend.pdf',
                                  title='Train and Test loss trend',
                                  xlabel='epoch',
                                  ylabel='loss',
                                  labels=['Train', 'Test'])
    
    # 使用训练好的模型进行预测，并保存模型的预测结果和模型结构。
    x, y = dataset[:, :n_var], dataset[:, -1]
    layers = [x] + predict(reg_model, x)
    io.save_layers(layers, save_dir=result_dir)
    io.save_nn_model(reg_model, savepath=f'{result_dir}nn_module.pt', save_type='dict')

    # 将数据集的变量和模型预测结果投影到二维空间，并保存为PDF文件，以便于比较真实数据和模型预测。
    draw.project_to_2d_and_save(vars=tuple([dataset[:, i] for i in range(n_var)]), zs=(y, layers[-1]),
                                savefile=f'{result_dir}compare.pdf',
                                zs_legends=['true', 'nn'])

    # draw extrapolate and interpolate data
    # 使用模型进行数据的外推和内插，并绘制结果曲线图。这里使用了数据集特定的映射函数和范围。
    x_ranges = INTER_MAP[data_filename]
    original_func = FUNC_MAP[data_filename]

    draw.draw_polate_data_curves(x_ranges=x_ranges,
                                 models=[reg_model.cpu()],
                                 original_func=original_func,
                                 model_labels=['nn'],
                                 savepath=f'{result_dir}polate.pdf',
                                 title='interpolate and extrapolate')

    # Save hyperparameters
    # 最后，将训练过程中的超参数保存为JSON文件，以便于后续的分析和记录。
    log_dict = {
        'neurons': list([layer.shape[1] for layer in layers]),
        'n_epoch': n_epoch,
        'batch_size': batch_size,
        'optimizer': optimizer.__class__.__name__,
        'lr': lr,
        'mlp_model': reg_model.__class__.__name__
    }
    with open(f'{result_dir}settings.json', 'w') as f:
        json.dump(log_dict, f, indent=4)

    # 这段代码的主要目的是训练一个神经网络模型，并对其性能进行评估和可视化。通过这个过程，可以了解模型在给定数据集上的表现，并调整超参数来优化模型。





