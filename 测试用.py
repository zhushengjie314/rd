import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
import random

start_infer = False
infer_all_cnn_output = []

def extract_submatrices_from_tensor(tensor, a):
    k, m, n = tensor.size()
    if a > m or a > n:
        raise ValueError("a 阶子矩阵的大小不能超过矩阵的行数或列数。")

    submatrices_list = []
    for i in range(k):
        random_number = random.sample(range(0, m-a+1), 2)#得到两个随机数

        submatrix1 = tensor[i, random_number[0]:random_number[0]+a, random_number[0]:random_number[0]+a]#随机抽取矩阵
        submatrix2 = tensor[i, random_number[1]:random_number[1]+a, random_number[1]:random_number[1]+a]
        submatrices_list.append(submatrix1)
        submatrices_list.append(submatrix2)

    combined_tensor = torch.stack(submatrices_list, dim=0)
    combined_tensor = combined_tensor.view(2 * k, a, a)
    return combined_tensor

"""以上函数的举例
tensor = torch.randn((1,3,3))
print(tensor)
result = extract_submatrices_from_tensor(tensor,2)
print(result)
print(result.shape)

>>
tensor([[[-1.0801,  0.4544, -0.2936],
         [-1.2573, -0.9693,  0.8195],
         [ 0.8219, -0.0355, -0.2532]]])
         
tensor([[[-1.0801,  0.4544],
         [-1.2573, -0.9693]],

        [[-0.9693,  0.8195],
         [-0.0355, -0.2532]]])
torch.Size([2, 2, 2])

"""



def get_vector(matrix):      #得到一个层所以矩阵所合并的特征向量
    sum = []
    for j in range(matrix.shape[0]):
        eigenvalues, eigenvectors = torch.linalg.eig(matrix[j])

        # 仅保留实数特征值对应的实数特征向量
        real_eigenvectors = []
        for i in range(len(eigenvalues)):
            if eigenvalues[i].imag == 0:
                real_eigenvectors.append(eigenvectors[:, i].real)

        if real_eigenvectors:
            part = torch.cat(real_eigenvectors)
            sum.append(part)
    if sum:
        sum = torch.cat(sum)
    else:
        sum = torch.zeros((1))
    return sum

"""
把矩阵特征值对应的特征向量拼接成一个向量
注意：如果输入该函数的matrix的shape是[2,3,3]即有两个[3,3]的矩阵，这个函数会把这两个3*3的矩阵的特征向量都拿出来并且拼接
"""


class CNNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNNModel, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3),  # 将 kernel_size 改为 100
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(16 * ((input_size - 3 + 1) // 2), 10),  # 更新线性层的输入大小
            nn.ReLU(),
            nn.Linear(10, output_size),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension (batch_size, channels, sequence_length)
        conv_out = self.conv_layer(x)
        conv_out = conv_out.view(x.size(0), -1)  # Flatten
        output = self.fc_layer(conv_out)
        return output










import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLSTMWithClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):  # Add num_classes parameter
        super(AttentionLSTMWithClassifier, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear_attention = nn.Linear(hidden_dim, 1)
        self.linear_classifier = nn.Linear(hidden_dim, num_classes)  # Use num_classes for the classifier output
        self.softmsx = torch.nn.Softmax(dim = 1)
    def forward(self, x):
        lstm_out, _ = self.lstm(x.unsqueeze(0))  # Add an extra dimension for batch size
        k = lstm_out[:, -1, :]
        output = self.linear_classifier(k)
        return output







class Random_Matrix(nn.Module):
    def __init__(self, M, k ,a):
        super(Random_Matrix, self).__init__()
        self.M = M
        self.k = k  #递减系数
        self.a = a  #子矩阵系数
        self.lstm = AttentionLSTMWithClassifier(40, 40, 1, 2)

          # 可调参数
        cnn_model_objects = []
        a = self.a
        c_output_size = 40
        for p in range(self.M + 1):
            if p == 0:
                c_input_size = 40000  # 参数
            else:
                c_input_size = (2 ** p) * (a ** 2)
                a = a - self.k
            cnn_model_objects.append(CNNModel(c_input_size, c_output_size))
        self.cnn = nn.ModuleList(cnn_model_objects)
    def forward(self, matrix):
        c_output_size = 40
        all_cnn_output = torch.zeros((self.M+1, c_output_size))
        a = self.a
        for j in range(self.M+1):
            #进行随机矩阵提取和cnn，lstm处理
            vector = get_vector(matrix)
            #把vector填充成符合的标准
            # 目标形状的大小
            if j == 0:
                target_size = 40000     #参数
            else:
                target_size = (2**j)*(a**2)
                a = a - self.k
            vector = torch.cat((vector, torch.zeros(target_size - len(vector))))
            matrix = extract_submatrices_from_tensor(matrix, a)
            output = self.cnn[j](vector.unsqueeze(0))
            all_cnn_output[j, :] = output.squeeze(0)     #
        value = self.lstm(all_cnn_output)
        if start_infer:
            print("forward_all_cnn_output",all_cnn_output)
            infer_all_cnn_output.append(all_cnn_output)
            print("forward_all_cnn_output_value",value)
        return value








def one_hot_encode(labels, num_classes):
    one_hot = torch.zeros(len(labels), num_classes)
    one_hot.scatter_(1, labels.view(-1, 1), 1)
    return one_hot







import torch
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成二分类数据集
X, y = make_classification(
    n_samples=200,   # 样本数量
    n_features=40000,     # 特征数量，这里为200x200=40000
    n_informative=10000,  # 有信息特征数量
    n_redundant=0,    # 冗余特征数量
    n_clusters_per_class=1,  # 每个类别中的簇数
    random_state=42   # 随机种子
)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.Tensor(X_train.reshape(160, 200, 200))
X_test_tensor = torch.Tensor(X_test.reshape(40, 200, 200))

y_train_tensor = torch.Tensor(y_train)
y_test_tensor = torch.Tensor(y_test)

#模型建立
model = Random_Matrix(4, 30, 160) #（第一个是分解次数， 第二个是递减量， 第三个是第一次分解的方阵的长）
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



sum_loss = torch.zeros((160))
for j in range(1):
    p = 0
    for i in range(160):
        inputs = X_train_tensor[i]
        num_classes = 2
        labels = one_hot_encode(torch.tensor([int(y_train_tensor[i])]), num_classes)
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs.unsqueeze(0))


        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sum_loss[p] = loss
        # print(output)
        p = p+1

        print(str(p)+'and'+str(loss))
    print(torch.mean(sum_loss))



# #模型训练
start_infer = True
all_test_output = []
for i in range(40):
    inputs = X_test_tensor[i]
    output = model(inputs.unsqueeze(0))
    print(output)
    predicted_class = torch.argmax(output).item()
    all_test_output.append(predicted_class)



all_test_output = torch.tensor([all_test_output])
print(all_test_output)

acu = torch.sum(all_test_output == y_test_tensor).item() / 580.0
acu_percent = acu * 100
print(acu_percent)

#计算cos相似度
for i in range(len(infer_all_cnn_output)):
    for j in range(len(infer_all_cnn_output)):
        x = infer_all_cnn_output[i]
        y = infer_all_cnn_output[j]
        similarity = F.cosine_similarity(x, y, dim=0)
        print(i, j, similarity)




