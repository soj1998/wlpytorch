import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tushare as ts
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy as copy
from torch.utils.data import DataLoader, TensorDataset


class GetData:
    def __init__(self, stock_id, save_path):
        self.stock_id = stock_id
        self.save_path=save_path
        self.data=None

    def getData(self):
        self.data = ts.get_hist_data(self.stock_id).iloc[::-1]
        self.data=self.data[["open", "close", "high", "low", "volume"]]
        self.close_min=self.data['volume'].min()
        self.close_max=self.data['volume'].max()
        self.data = self.data.apply(lambda x:(x-min(x))/(max(x)-min(x)))
        self.data.to_csv(self.save_path)
        return self.data

    def process_data(self, n):
        if self.data is None:
            self.getData()
        feature=[self.data.iloc[i:i+n].values.tolist() for i in range(len(self.data)-n+2)
                 if i + n < len(self.data)]
        label = [self.data.close.values[i+n] for i in range(len(self.data)-n+2)
                 if i + n < len(self.data)]
        train_x = feature[:500]
        test_x = feature[500:]
        train_y = label[:500]
        test_y = label[500:]
        return train_x, test_x, train_y, test_y


class Model(nn.Module):
    def __init__(self, n):
        super(Model, self).__init__()
        self.lstm_layer = nn.LSTM(input_size=n, hidden_size=256, batch_first=True)
        self.linear_layer = nn.Linear(in_features=256, out_features=1, bias=True)

    def forward(self, x):
        out1, (h_n, h_c) = self.lstm_layer(x)
        a, b, c = h_n.shape
        out2 = self.linear_layer(h_n.reshape(a * b, c))
        return out2


def train_model(epoch, train_dataLoader, test_dataLoadter):
    best_model = None
    train_loss = 0
    test_loss = 0
    best_loss = 100
    epoch_cnt = 0
    for _ in range(epoch):
        total_train_loss = 0
        total_train_num = 0
        total_test_loss = 0
        total_test_num = 0
        for x, y in tqdm(train_dataLoader,
                         desc='Epoch:()|Train Loss:{}|Test Loss:{}'
                                 .format(_, train_loss, test_loss)):
            x_num = len(x)
            p = model(x)
            loss = loss_func(p, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            total_test_num += x_num
            test_loss = total_test_loss / total_test_num
        if best_loss > test_loss:
            best_loss = test_loss
            best_model = copy(model)
            epoch_cnt = 0
        else:
            epoch_cnt += 1

        if epoch_cnt > early_stop:
            torch.save(best_model.state_dict(), './lstm_.pth')
            break


def test_model(test_dataLoader_):
    pred = []
    label = []
    model_ = Model(5)
    model_.load_state_dict(torch.load("./lstm_.pth"))
    model_.eval()
    total_test_loss = 0
    total_test_num = 0
    for x, y in test_dataLoader_:
        x_num = len(x)
        p = model_(x)
        loss = loss_func(p, y)
        total_test_loss += loss.item()
        total_test_num += x_num
        pred.extend(p.data.squeeze(1).tolist())
        label.extend(y.tolist())
    test_loss = total_test_loss / total_test_num
    return pred, label, test_loss


def plot_img(data,pred):
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.figure(figsize=(12,7))
    plt.plot(range(len(pred)),pred,color='green')
    plt.plot(range(len(data)),data,color='blue')
    for i in range(0, len(pred) - 3, 5):
        price=[data[i]+pred[j]-pred[i] for j in range(i, i+3)]
        plt.plot(range(i,i+3), price, color='red')
    plt.xticks(fontproperties='Times New Roman', size=15)
    plt.yticks(fontproperties='Times New Roman', size=15)
    plt.xlabel('日期', fontsize=18)
    plt.ylabel('成交量', fontsize=18)
    plt.show()


if __name__ =='__main__':
    #超参数
    days_num = 5
    epoch =20
    fea = 5
    batch_size= 20
    early_stop = 5
    #初始化模型
    model = Model(fea)
    #数据处理
    GD = GetData(stock_id='601398', save_path='./data.csv')
    x_train, x_test, y_train, y_test = GD.process_data(days_num)
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()
    train_data = TensorDataset(x_train, y_train)
    train_dataLoader = DataLoader(train_data, batch_size=batch_size)
    test_data = TensorDataset(x_test, y_test)
    test_dataLoader = DataLoader(test_data, batch_size=batch_size)
    # 损失函数、优化器
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(epoch, train_dataLoader, test_dataLoader)
    p, y, test_loss = test_model(test_dataLoader)
    # 绘制折线图
    pred = [ele * (GD.close_max - GD.close_min) + GD.close_min for ele in p]
    data = [ele * (GD.close_max - GD.close_min) + GD.close_min for ele in y]
    plot_img(data, pred)
    # 输出模型损失
    print('模型损失：', test_loss)
