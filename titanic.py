import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import csv

# 导入数据集
train_filepath = r"train.csv"
train_data = pd.read_csv(train_filepath)
test_filepath = r"test.csv"
test_data = pd.read_csv(test_filepath)

# 训练数据预处理
def PreprocessTrainData(all_pf):
    # 预处理1：去掉无关特征
    cols = ["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    all_pf = all_pf[cols]
    # 预处理2：填充确实特征并标准化
    age_mean = all_pf["Age"].mean()
    all_pf["Age"] = all_pf["Age"].fillna(age_mean)

    fare_mean = all_pf["Fare"].mean()
    all_pf["Fare"] = all_pf["Fare"].fillna(fare_mean)

    # 预处理3：性别编码
    all_pf["Sex"] = all_pf["Sex"].map({"female": 0, "male": 1}).astype(int)

    # 预处理4：登港地点转换为one-hot编码
    x_OneHot_df = pd.get_dummies(data=all_pf, columns=["Embarked"])
    ndarray = x_OneHot_df.values

    # 预处理5：全体特征标准化，标签向量化
    label = ndarray[:, 0]
    label = label.reshape(label.shape[0], 1)
    features = ndarray[:, 1:]
    mean = features.mean(axis=0)
    features -= mean
    std = features.std(axis=0)
    features /= std

    return label, features


# 测试数据预处理
def PreprocessTestData(all_df):
    # 预处理1：筛除无关特征
    cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    all_df = all_df[cols]

    # 预处理2：填充缺失特征并标准化特征
    age_mean = all_df["Age"].mean()
    all_df["Age"] = all_df["Age"].fillna(age_mean)

    fare_mean = all_df["Fare"].mean()
    all_df["Fare"] = all_df["Fare"].fillna(fare_mean)

    # 预处理3：性别编码0-1
    all_df["Sex"] = all_df["Sex"].map({"female": 0, "male": 1}).astype(int)

    # 预处理4：登港地点转换为one-hot编码
    x_OneHot_df = pd.get_dummies(data=all_df, columns=["Embarked"])
    ndarray = x_OneHot_df.values

    # 预处理5：全体特征标准化，标签向量化
    features = ndarray
    mean = features.mean(axis=0)
    features -= mean
    std = features.std(axis=0)
    features /= std
    return features


y_train, x_train = PreprocessTrainData(train_data)
x_train_tenser = torch.from_numpy(x_train)
y_train_tenser = torch.from_numpy(y_train)
x_test = PreprocessTestData(test_data)
x_test_tenser = torch.from_numpy(x_test)


# 留出验证集
num_val = 300
np.random.shuffle([x_train, y_train])
x_val = x_train[:num_val]
x_val_tensor = torch.from_numpy((x_val))
partial_x_train = x_train[num_val:]

y_val = y_train[:num_val]
partial_y_train = y_train[num_val:]

# 构造网络模型
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.linear1 = torch.nn.Linear(9, 6)
        self.linear2 = torch.nn.Linear(6, 3)
        self.linear3 = torch.nn.Linear(3, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = model()

# 构造优化器和损失函数
criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


cost_list = []
# 开始训练
for epoch in range(1000):
    # 正向传播
    y_pred = model(x_train_tenser.float())
    loss = criterion(y_pred, y_train_tenser.float())
    cost_list.append(loss.item())
    print("epoch:", epoch, "cost:", loss.item())
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    # 更新参数
    optimizer.step()

y_pred = model(x_test_tenser.float())
y_pred = np.array(y_pred.data)
for i in range(y_pred.shape[0]):
    if y_pred[i, 0] > 0.5:
        y_pred[i, 0] = 1
    else:
        y_pred[i, 0] = 0
with open(r"gender_submission.csv", 'w+', newline='') as f:
    csv_file = csv.writer(f)
    csv_file.writerows(y_pred)

plt.plot(cost_list)
plt.xlabel("epoch")
plt.ylabel("cost")
plt.show()



