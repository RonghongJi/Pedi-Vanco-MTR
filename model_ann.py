# -*- coding: utf-8 -*-
"""
Created on 2026/04/24 13:50:00

@File    :   model_ann.py
@Author  :   Ronghong Ji
"""

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import random

seed = 486

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 自定义 Swish 激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# 定义神经网络模型
class ANN(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation='relu', output_dim=2):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = self.get_activation_function(activation)

    def get_activation_function(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'selu':
            return nn.SELU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        elif activation == 'swish':
            return Swish()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义训练过程
def train_model(model, criterion, optimizer, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

# 定义评估指标函数
def calculate_regression_metrics(y_true, y_pred):
    metrics = {}
    for i, column in enumerate(['CL', 'V']):
        metrics[column] = {
            'RMSE': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            'R2': r2_score(y_true[:, i], y_pred[:, i]),
            'MAE': mean_absolute_error(y_true[:, i], y_pred[:, i]),
        }
    return metrics

# 十折交叉验证部分
def cross_val_regression_metrics(model_class, param_grid, X, y, num_epochs, cv):
    metrics_list = {col: [] for col in ['CL', 'V']}
    for train_idx, val_idx in cv.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # 数据加载
        train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
        target_tensor = torch.tensor(y_train_fold, dtype=torch.float32)
        val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        val_target_tensor = torch.tensor(y_val_fold, dtype=torch.float32)

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_tensor, target_tensor), batch_size=32, shuffle=True)

        # 模型参数
        input_dim = X.shape[1]
        model = model_class(input_dim=input_dim, output_dim=2, **param_grid)  # 传递 output_dim
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # 训练模型
        train_model(model, criterion, optimizer, train_loader, num_epochs)

        # 验证模型
        model.eval()
        with torch.no_grad():
            val_predictions = model(val_tensor).numpy()
            fold_metrics = calculate_regression_metrics(y_val_fold, val_predictions)
            for col in ['CL', 'V']:
                metrics_list[col].append(fold_metrics[col])

    # 计算平均值和标准差
    final_metrics = {}
    for col in ['CL', 'V']:
        final_metrics[col] = {
            'RMSE': (np.mean([m['RMSE'] for m in metrics_list[col]]), np.std([m['RMSE'] for m in metrics_list[col]])),
            'R2': (np.mean([m['R2'] for m in metrics_list[col]]), np.std([m['R2'] for m in metrics_list[col]])),
            'MAE': (np.mean([m['MAE'] for m in metrics_list[col]]), np.std([m['MAE'] for m in metrics_list[col]])),
        }
    return final_metrics

# 定义网格搜索
def grid_search_cv(model_class, param_grid, X, y, num_epochs, cv):
    best_metrics = None
    best_params = {}
    
    for hidden_dim in param_grid['hidden_dim']:
        for activation in param_grid['activation']:
            print(f"Optimizing hidden_dim={hidden_dim}, activation={activation}...")
            params = {'hidden_dim': hidden_dim, 'activation': activation}
            metrics = cross_val_regression_metrics(model_class, params, X, y, num_epochs, cv)

            # 根据 R2 选择最佳参数
            if best_metrics is None or metrics['CL']['R2'][0] > best_metrics['CL']['R2'][0]:
                best_metrics = metrics
                best_params = params

    return best_params, best_metrics

# 加载数据
df = pd.read_excel(r'../data/df_rf_log_allfeature.xlsx')

# 特征列，去除 'CL' 和 'V' 列
X = df.drop(columns=['CL', 'V']).values
y = df[['CL', 'V']].values

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# 定义参数网格
param_grids = {
    'hidden_dim': [16, 32, 64, 128, 256],
    'activation': ['relu', 'selu', 'elu', 'leaky_relu', 'swish']
}

# 定义十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# 网格搜索获得最佳参数
best_params, best_metrics = grid_search_cv(ANN, param_grids, X_train, y_train, num_epochs=100, cv=kf)

# 使用最佳参数训练最终模型
final_model = ANN(input_dim=X.shape[1], hidden_dim=best_params['hidden_dim'], activation=best_params['activation'], output_dim=2)
train_tensor = torch.tensor(X_train, dtype=torch.float32)
target_tensor = torch.tensor(y_train, dtype=torch.float32)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_tensor, target_tensor), batch_size=32, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr=0.01)
train_model(final_model, criterion, optimizer, train_loader, num_epochs=100)

# 测试集预测
final_model.eval()
with torch.no_grad():
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_pred = final_model(test_tensor).numpy()

# 计算测试集上的回归指标
test_metrics = calculate_regression_metrics(y_test, y_pred)

# 合并十折交叉验证和测试集的回归指标
metrics = {}
for col in ['CL', 'V']:
    metrics[f'Ten_fold_CV_RMSE_{col}'] = f"{best_metrics[col]['RMSE'][0]:.2f} ± {best_metrics[col]['RMSE'][1]:.2f}"
    metrics[f'Ten_fold_CV_R2_{col}'] = f"{best_metrics[col]['R2'][0]:.2f} ± {best_metrics[col]['R2'][1]:.2f}"
    metrics[f'Ten_fold_CV_MAE_{col}'] = f"{best_metrics[col]['MAE'][0]:.2f} ± {best_metrics[col]['MAE'][1]:.2f}"

# 添加测试集指标
for col in ['CL', 'V']:
    metrics[f'Test_RMSE_{col}'] = f"{test_metrics[col]['RMSE']:.2f}"
    metrics[f'Test_R2_{col}'] = f"{test_metrics[col]['R2']:.2f}"
    metrics[f'Test_MAE_{col}'] = f"{test_metrics[col]['MAE']:.2f}"

# 保存到 XLSX 文件
metrics_df = pd.DataFrame(metrics, index=[0])
metrics_df.to_excel(r'../output/ML输出/ANN_GridSearch_Regression.xlsx', index=False)
metrics_df

# %%
best_params

# %%
import pandas as pd

# 如果 y_test 是 pandas DataFrame 或 Series，可以重置索引以避免问题
y_test_df = pd.DataFrame(y_test).reset_index(drop=True)  # 如果是 Series 或 DataFrame
y_pred_df = pd.DataFrame(y_pred).reset_index(drop=True)  # 同样处理预测值

# 确保列名称正确
y_test_df.columns = ['True Values cl', 'True Values v']
y_pred_df.columns = ['Predicted Values cl', 'Predicted Values v']

# 将两者合并到一个 DataFrame 中
output_df = pd.concat([y_test_df, y_pred_df], axis=1)

# 保存到 Excel 文件
output_df.to_excel(r'../output/pred/ANN_predictions.xlsx', index=False)

print("ok!")


# %%
import numpy as np
import pandas as pd

def calculate_accuracy_within_percentage(y_true, y_pred, percentage):
    # 计算范围
    lower_bound = y_true * (1 - percentage / 100)
    upper_bound = y_true * (1 + percentage / 100)
    
    # 检查预测值是否在范围内
    within_range = (y_pred >= lower_bound) & (y_pred <= upper_bound)
    
    return np.mean(within_range)  # 返回在范围内的比例

# 计算在不同百分比误差范围内的准确率
accuracies = {}
for pct in [10, 20, 30, 40, 50]:
    accuracies[f"Accuracy_within_{pct}%_CL"] = calculate_accuracy_within_percentage(y_test[:,0], y_pred[:, 0], pct)
    accuracies[f"Accuracy_within_{pct}%_V"] = calculate_accuracy_within_percentage(y_test[:,1], y_pred[:, 1], pct)

# 创建一个新的 DataFrame 用于存储准确率
accuracy_df = pd.DataFrame({
    '±10%': [accuracies['Accuracy_within_10%_CL'], accuracies['Accuracy_within_10%_V']],
    '±20%': [accuracies['Accuracy_within_20%_CL'], accuracies['Accuracy_within_20%_V']],
    '±30%': [accuracies['Accuracy_within_30%_CL'], accuracies['Accuracy_within_30%_V']],
    '±40%': [accuracies['Accuracy_within_40%_CL'], accuracies['Accuracy_within_40%_V']],
    '±50%': [accuracies['Accuracy_within_50%_CL'], accuracies['Accuracy_within_50%_V']],
}, index=['CL', 'V'])

# 显示准确率 DataFrame
print(accuracy_df)

# 保存到 XLSX 文件
accuracy_df.to_excel(r'../output/accuracy/ANN_acc.xlsx', index=False)


# %%
import matplotlib.pyplot as plt

# 绘制拟合散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test[:,0], y_pred[:, 0], color='black', label='CL', s=15)
plt.plot([y_test[:,0].min(), y_test[:,0].max()], [y_test[:,0].min(), y_test[:,0].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Fitted Scatter Plot for CL')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test[:,1], y_pred[:, 1], color='black', label='V', s=15)
plt.plot([y_test[:,1].min(), y_test[:,1].max()], [y_test[:,1].min(), y_test[:,1].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Fitted Scatter Plot for V')
plt.legend()
plt.grid()
plt.show()

# 绘制残差图
plt.figure(figsize=(10, 6))
residuals_cl = y_test[:,0] - y_pred[:, 0]
plt.scatter(y_test[:,0], residuals_cl, color='black', label='Residuals CL', s=15)
plt.axhline(0, color='black', linestyle='--')  # 0参考线
plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for CL')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
residuals_v = y_test[:,1] - y_pred[:, 1]
plt.scatter(y_test[:,1], residuals_v, color='black', label='Residuals V', s=15)
plt.axhline(0, color='black', linestyle='--')  # 0参考线
plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.title('Residual Plot for V')
plt.legend()
plt.grid()
plt.show()



