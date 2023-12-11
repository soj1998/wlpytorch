import torch
import torch.nn
import torch.utils.data as Data
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

x = torch.unsqueeze(torch.linspace(-1, 1, 500), dim=1)
y = x.pow(3)
LR = 0.01
batch_size = 15
epoches = 5
torch.manual_seed(10)
dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2)


class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden_layer = torch.nn.Linear(n_input, n_hidden)
        self.output_layer = torch.nn.Linear(n_hidden, n_output)

    def forward(self, input):
        x = torch.relu(self.hidden_layer(input))
        output = self.output_layer(x)
        return output


def train():
    net_SGD = Net(1, 10, 1)
    net_Momentum = Net(1, 10, 1)
    net_AdaGrad = Net(1, 10, 1)
    net_RMSprop = Net(1, 10, 1)
    net_Adam = Net(1, 10, 1)
    nets = [net_SGD, net_Momentum, net_AdaGrad, net_RMSprop, net_Adam]
    # 定义优化器
    optimizer_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    optimizer_Momentum = torch.optim.SGD(net_Momentum.parameters(),
                                         lr=LR, momentum=0.6)
    optimizer_AdaGrad = torch.optim.Adagrad(net_AdaGrad.parameters(),
                                            lr=LR, lr_decay=0)
    optimizer_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(),
                                            lr=LR, alpha=0.9)
    optimizer_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR,
                                      betas=(0.9, 0.99))
    optimizers = [optimizer_SGD, optimizer_Momentum, optimizer_AdaGrad,
                  optimizer_RMSprop, optimizer_Adam]
    # 定义损失函数
    loss_function = torch.nn.MSELoss()
    losses = [[], [], [], [], []]
    for epoch in range(epoches):
        for step, (batch_x, batch_y) in enumerate(loader):
            for net, optimizer, loss_list in zip(nets, optimizers, losses):
                pred_y = net(batch_x)
                loss = loss_function(pred_y, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_list.append(loss.data.numpy())
    plt.figure(figsize=(12, 7))
    labes = ['SGD', 'Momentum', 'AdaGrad', 'RMSprop', 'Adam']
    for i, loss in enumerate(losses):
        plt.plot(loss, label=labes[i])
    plt.legend(loc='upper right', fontsize=15)
    plt.tick_params(labelsize=13)
    plt.xlabel('训练步骤', size=15)
    plt.ylabel('模型损失', size=15)
    plt.ylim((0, 0.3))
    plt.show()


if __name__ == "__main__":
    train()

