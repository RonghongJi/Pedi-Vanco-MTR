# -*- coding: utf-8 -*-
"""
Created on 2026/04/24 14:07:38

@File    :   model_resnet.py
@Author  :   Ronghong Ji
"""

# %%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random

seed=486

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# 定义一维残差块，用于表格数据
class ResidualBlock1D(nn.Module):
    def __init__(self, in_features, out_features, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += self.shortcut(x)
        return self.relu(out)

# 定义 ResNet 模型
class ResNet1D(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 128, 256, 512], activation='relu', output_dim=2):
        super(ResNet1D, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.activation = self.get_activation_function(activation)
        
        self.layer1 = self._make_layer(hidden_dims[0], hidden_dims[1])
        self.layer2 = self._make_layer(hidden_dims[1], hidden_dims[2])
        self.layer3 = self._make_layer(hidden_dims[2], hidden_dims[3])
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(hidden_dims[3], output_dim)

    def get_activation_function(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'selu':
            return nn.SELU()
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def _make_layer(self, in_features, out_features):
        layers = [ResidualBlock1D(in_features, out_features),
                  ResidualBlock1D(out_features, out_features)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.activation(self.bn1(self.fc1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x.unsqueeze(-1)).squeeze(-1)
        return self.fc_out(x)

# 训练函数
def train_model(model, criterion, optimizer, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

# 计算回归指标
def calculate_regression_metrics(y_true, y_pred):
    metrics = {}
    for i, column in enumerate(['CL', 'V']):
        metrics[column] = {
            'RMSE': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
            'R2': r2_score(y_true[:, i], y_pred[:, i]),
            'MAE': mean_absolute_error(y_true[:, i], y_pred[:, i]),
        }
    return metrics

# 十折交叉验证
def cross_val_regression_metrics(model_class, param_grid, X, y, num_epochs, cv):
    metrics_list = {col: [] for col in ['CL', 'V']}
    for train_idx, val_idx in cv.split(X):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        train_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
        target_tensor = torch.tensor(y_train_fold, dtype=torch.float32)
        val_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        val_target_tensor = torch.tensor(y_val_fold, dtype=torch.float32)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_tensor, target_tensor), batch_size=32, shuffle=True)

        input_dim = X.shape[1]
        model = model_class(input_dim=input_dim, output_dim=2, **param_grid)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        train_model(model, criterion, optimizer, train_loader, num_epochs)

        model.eval()
        with torch.no_grad():
            val_predictions = model(val_tensor).numpy()
            fold_metrics = calculate_regression_metrics(y_val_fold, val_predictions)
            for col in ['CL', 'V']:
                metrics_list[col].append(fold_metrics[col])

    final_metrics = {}
    for col in ['CL', 'V']:
        final_metrics[col] = {
            'RMSE': (np.mean([m['RMSE'] for m in metrics_list[col]]), np.std([m['RMSE'] for m in metrics_list[col]])),
            'R2': (np.mean([m['R2'] for m in metrics_list[col]]), np.std([m['R2'] for m in metrics_list[col]])),
            'MAE': (np.mean([m['MAE'] for m in metrics_list[col]]), np.std([m['MAE'] for m in metrics_list[col]])),
        }
    return final_metrics

def grid_search_cv(model_class, param_grid, X, y, num_epochs, cv):
    best_metrics = None
    best_params = {}
    
    for hidden_dims in param_grid['hidden_dims']:
        for activation in param_grid['activation']:
            print(f"Optimizing hidden_dims={hidden_dims}, activation={activation}...")
            params = {'hidden_dims': hidden_dims, 'activation': activation}
            metrics = cross_val_regression_metrics(model_class, params, X, y, num_epochs, cv)

            if best_metrics is None or metrics['CL']['R2'][0] > best_metrics['CL']['R2'][0]:
                best_metrics = metrics
                best_params = params

    return best_params, best_metrics

# 定义数据和参数
df = pd.read_excel(r'../data/df_rf_log_allfeature.xlsx')
X = df.drop(columns=['CL', 'V']).values
y = df[['CL', 'V']].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

param_grids = {
    'hidden_dims': [[64, 128, 256, 512], [16, 32, 64, 128], [32, 64, 128, 256]], # 设置隐藏层维度
    'activation': ['relu', 'selu', 'elu', 'leaky_relu'],
}
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# 交叉验证并选择最佳参数
best_params, best_metrics = grid_search_cv(ResNet1D, param_grids, X_train, y_train, num_epochs=100, cv=kf)

# 使用最佳参数训练最终模型
final_model = ResNet1D(input_dim=X.shape[1], output_dim=2, hidden_dims=best_params['hidden_dims'], 
                        activation=best_params['activation'])
train_tensor = torch.tensor(X_train, dtype=torch.float32)
target_tensor = torch.tensor(y_train, dtype=torch.float32)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_tensor, target_tensor), batch_size=32, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(final_model.parameters(), lr=0.01)
train_model(final_model, criterion, optimizer, train_loader, num_epochs=100)

# 测试集预测和评估
final_model.eval()
with torch.no_grad():
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_pred = final_model(test_tensor).numpy()
test_metrics = calculate_regression_metrics(y_test, y_pred)

# 保存结果
metrics = {}
for col in ['CL', 'V']:
    metrics[f'Ten_fold_CV_RMSE_{col}'] = f"{best_metrics[col]['RMSE'][0]:.2f} ± {best_metrics[col]['RMSE'][1]:.2f}"
    metrics[f'Ten_fold_CV_R2_{col}'] = f"{best_metrics[col]['R2'][0]:.2f} ± {best_metrics[col]['R2'][1]:.2f}"
    metrics[f'Ten_fold_CV_MAE_{col}'] = f"{best_metrics[col]['MAE'][0]:.2f} ± {best_metrics[col]['MAE'][1]:.2f}"

for col in ['CL', 'V']:
    metrics[f'Test_RMSE_{col}'] = f"{test_metrics[col]['RMSE']:.2f}"
    metrics[f'Test_R2_{col}'] = f"{test_metrics[col]['R2']:.2f}"
    metrics[f'Test_MAE_{col}'] = f"{test_metrics[col]['MAE']:.2f}"

metrics_df = pd.DataFrame(metrics, index=[0])
metrics_df.to_excel(r'../output/ML输出/ResNet_GridSearch_Regression.xlsx', index=False)
metrics_df
# %%
# 保存最佳参数到 Excel 文件
# 展平列表并转换为 DataFrame
best_params_df = pd.DataFrame([best_params])
best_params_df.to_excel(r'../output/param/ResNet_best_params.xlsx', index=False)
best_params_df

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
output_df.to_excel(r'../output/pred/ResNet_predictions.xlsx', index=False)

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
accuracy_df.to_excel(r'../output/accuracy/ResNet_acc.xlsx', index=False)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 绘制 CL 拟合散点图并添加 R²
plt.figure(figsize=(10, 6))
plt.scatter(y_test[:,0], y_pred[:, 0], color='black', s=15)
plt.plot([y_test[:,0].min(), y_test[:,0].max()], [y_test[:,0].min(), y_test[:,0].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('ResNet Scatter Plot for CL', fontsize=20)

# 计算 R² 并添加到图中
r2_cl = r2_score(y_test[:,0], y_pred[:, 0])
plt.text(0.8, 0.3, f"R² = {r2_cl:.2f}", fontsize=16, fontweight='bold', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.show()

# 绘制 V 拟合散点图并添加 R²
plt.figure(figsize=(10, 6))
plt.scatter(y_test[:,1], y_pred[:, 1], color='black', s=15)
plt.plot([y_test[:,1].min(), y_test[:,1].max()], [y_test[:,1].min(), y_test[:,1].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('ResNet Scatter Plot for V', fontsize=20)

# 计算 R² 并添加到图中
r2_v = r2_score(y_test[:,1], y_pred[:, 1])
plt.text(0.8, 0.3, f"R² = {r2_v:.2f}", fontsize=16, fontweight='bold', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.show()



# %%
import matplotlib.pyplot as plt
# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 绘制拟合散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test[:,0], y_pred[:, 0], color='black', s=15)
plt.plot([y_test[:,0].min(), y_test[:,0].max()], [y_test[:,0].min(), y_test[:,0].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('ResNet Fitted Scatter Plot for CL', fontsize=20)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test[:,1], y_pred[:, 1], color='black', s=15)
plt.plot([y_test[:,1].min(), y_test[:,1].max()], [y_test[:,1].min(), y_test[:,1].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('ResNet Fitted Scatter Plot for V', fontsize=20)
plt.show()

# 绘制残差图
plt.figure(figsize=(10, 6))
residuals_cl = y_test[:,0] - y_pred[:, 0]
plt.scatter(y_test[:,0], residuals_cl, color='black', s=15)
plt.axhline(0, color='black', linestyle='--')  # 0参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Residuals', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('ResNet Residual Plot for CL', fontsize=20)
plt.show()

plt.figure(figsize=(10, 6))
residuals_v = y_test[:,1] - y_pred[:, 1]
plt.scatter(y_test[:,1], residuals_v, color='black', s=15)
plt.axhline(0, color='black', linestyle='--')  # 0参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Residuals', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('ResNet Residual Plot for V', fontsize=20)
plt.show()



