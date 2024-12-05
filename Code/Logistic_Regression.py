import torch

# 定义超参数
num_class = 4
input_size = 4
hidden_size = 8
embedding_size = 10
num_layers = 2
batch_size = 1
seq_len = 5

# 数据
idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]
inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)

# 调整输入数据形状
inputs = inputs.unsqueeze(0)  # 添加 batch 维度，变为 (batch_size, seq_len)

# 模型定义
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        # 初始化 hidden，维度应为 (num_layers, batch_size, hidden_size)
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)  # (batch_size, seq_len, embedding_size)
        x, _ = self.rnn(x, hidden)  # (batch_size, seq_len, hidden_size)
        x = self.fc(x)  # (batch_size, seq_len, num_class)
        return x.view(-1, num_class)  # 展平为 (batch_size * seq_len, num_class)

# 初始化模型
net = Model()

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

# 训练循环
for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)  # 前向传播
    loss = criterion(outputs, labels)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 参数更新

    # 打印预测结果
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))
