# -*- coding: utf-8 -*-
"""
Created on 2026/04/24 14:12:51

@File    :   model_linerregression.py
@Author  :   Ronghong Ji
"""

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

seed = 486

# 加载数据
df = pd.read_excel(r'../df_rf_log_allfeature.xlsx')

# 特征列，去除 'CL' 和 'V' 列
X = df.drop(columns=['CL', 'V'])
y = df[['CL', 'V']]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# 使用多输出线性回归模型
linear_regression_model = MultiOutputRegressor(LinearRegression())

# 定义评估指标函数，分别计算 CL 和 V 的指标
def calculate_regression_metrics(y_true, y_pred):
    metrics = {}
    for i, column in enumerate(y_true.columns):
        metrics[column] = {
            'RMSE': np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i])),
            'R2': r2_score(y_true.iloc[:, i], y_pred[:, i]),
            'MAE': mean_absolute_error(y_true.iloc[:, i], y_pred[:, i]),
        }
    return metrics

# 十折交叉验证部分，返回 CL 和 V 的回归指标
def cross_val_regression_metrics(model, X, y, cv):
    metrics_list = {col: [] for col in y.columns}
    for train_idx, val_idx in cv.split(X):
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train_fold, y_train_fold)
        val_predictions = model.predict(X_val_fold)
        fold_metrics = calculate_regression_metrics(y_val_fold, val_predictions)
        for col in y.columns:
            metrics_list[col].append(fold_metrics[col])
    
    # 计算平均值和标准差
    final_metrics = {}
    for col in y.columns:
        final_metrics[col] = {
            'RMSE': (np.mean([m['RMSE'] for m in metrics_list[col]]), np.std([m['RMSE'] for m in metrics_list[col]])),
            'R2': (np.mean([m['R2'] for m in metrics_list[col]]), np.std([m['R2'] for m in metrics_list[col]])),
            'MAE': (np.mean([m['MAE'] for m in metrics_list[col]]), np.std([m['MAE'] for m in metrics_list[col]])),
        }
    return final_metrics

# 定义十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# 计算十折交叉验证的回归指标
linear_reg_cv_metrics = cross_val_regression_metrics(linear_regression_model, X_train, y_train, kf)

# 训练最终模型并预测
linear_regression_model.fit(X_train, y_train)
linear_reg_y_pred = linear_regression_model.predict(X_test)

# 计算测试集上的回归指标
linear_reg_test_metrics = calculate_regression_metrics(y_test, linear_reg_y_pred)

# 合并十折交叉验证和测试集的回归指标
metrics = {}
for col in y_train.columns:
    metrics[f'Ten_fold_CV_RMSE_{col}'] = f"{linear_reg_cv_metrics[col]['RMSE'][0]:.2f} ± {linear_reg_cv_metrics[col]['RMSE'][1]:.2f}"
    metrics[f'Ten_fold_CV_R2_{col}'] = f"{linear_reg_cv_metrics[col]['R2'][0]:.2f} ± {linear_reg_cv_metrics[col]['R2'][1]:.2f}"
    metrics[f'Ten_fold_CV_MAE_{col}'] = f"{linear_reg_cv_metrics[col]['MAE'][0]:.2f} ± {linear_reg_cv_metrics[col]['MAE'][1]:.2f}"

# 添加测试集指标
for col in y_test.columns:
    metrics[f'Test_RMSE_{col}'] = f"{linear_reg_test_metrics[col]['RMSE']:.2f}"
    metrics[f'Test_R2_{col}'] = f"{linear_reg_test_metrics[col]['R2']:.2f}"
    metrics[f'Test_MAE_{col}'] = f"{linear_reg_test_metrics[col]['MAE']:.2f}"

# 保存到 XLSX 文件
metrics_df = pd.DataFrame(metrics, index=[0])
metrics_df.to_excel(r'../ML输出/LinearRegression_GridSearch_Regression.xlsx', index=False)
metrics_df

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
    accuracies[f"Accuracy_within_{pct}%_CL"] = calculate_accuracy_within_percentage(y_test['CL'].values, linear_reg_y_pred[:, 0], pct)
    accuracies[f"Accuracy_within_{pct}%_V"] = calculate_accuracy_within_percentage(y_test['V'].values, linear_reg_y_pred[:, 1], pct)

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
accuracy_df.to_excel(r'../accuracy/LinearRegression_acc.xlsx', index=False)

# %%
import pandas as pd

# 如果 y_test 是 pandas DataFrame 或 Series，可以重置索引以避免问题
y_test_df = pd.DataFrame(y_test).reset_index(drop=True)  # 如果是 Series 或 DataFrame
linear_reg_y_pred_df = pd.DataFrame(linear_reg_y_pred).reset_index(drop=True)  # 同样处理预测值

# 确保列名称正确
y_test_df.columns = ['True Values cl', 'True Values v']
linear_reg_y_pred_df.columns = ['Predicted Values cl', 'Predicted Values v']

# 将两者合并到一个 DataFrame 中
output_df = pd.concat([y_test_df, linear_reg_y_pred_df], axis=1)

# 保存到 Excel 文件
output_df.to_excel(r'../pred/LinearRegression_predictions.xlsx', index=False)

print("ok!")


# %%
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 绘制 CL 拟合散点图并添加 R²
plt.figure(figsize=(10, 6))
plt.scatter(y_test['CL'], linear_reg_y_pred[:, 0], color='black', s=15)
plt.plot([y_test['CL'].min(), y_test['CL'].max()], [y_test['CL'].min(), y_test['CL'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('LinearRegression Scatter Plot for CL', fontsize=20)

# 计算 R² 并添加到图中
r2_cl = r2_score(y_test['CL'], linear_reg_y_pred[:, 0])
plt.text(0.8, 0.3, f"R² = {r2_cl:.2f}", fontsize=16, fontweight='bold', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.show()

# 绘制 V 拟合散点图并添加 R²
plt.figure(figsize=(10, 6))
plt.scatter(y_test['V'], linear_reg_y_pred[:, 1], color='black', s=15)
plt.plot([y_test['V'].min(), y_test['V'].max()], [y_test['V'].min(), y_test['V'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('LinearRegression Scatter Plot for V', fontsize=20)

# 计算 R² 并添加到图中
r2_v = r2_score(y_test['V'], linear_reg_y_pred[:, 1])
plt.text(0.8, 0.3, f"R² = {r2_v:.2f}", fontsize=16, fontweight='bold', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.show()

# %%
import matplotlib.pyplot as plt
# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 绘制拟合散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test['CL'], linear_reg_y_pred[:, 0], color='black', s=15)
plt.plot([y_test['CL'].min(), y_test['CL'].max()], [y_test['CL'].min(), y_test['CL'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('LinearRegression Fitted Scatter Plot for CL', fontsize=20)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test['V'], linear_reg_y_pred[:, 1], color='black', s=15)
plt.plot([y_test['V'].min(), y_test['V'].max()], [y_test['V'].min(), y_test['V'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('LinearRegression Fitted Scatter Plot for V', fontsize=20)
plt.show()

# 绘制残差图
plt.figure(figsize=(10, 6))
residuals_cl = y_test['CL'] - linear_reg_y_pred[:, 0]
plt.scatter(y_test['CL'], residuals_cl, color='black', s=15)
plt.axhline(0, color='black', linestyle='--')  # 0参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Residuals', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('LinearRegression Residual Plot for CL', fontsize=20)
plt.show()

plt.figure(figsize=(10, 6))
residuals_v = y_test['V'] - linear_reg_y_pred[:, 1]
plt.scatter(y_test['V'], residuals_v, color='black', s=15)
plt.axhline(0, color='black', linestyle='--')  # 0参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Residuals', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('LinearRegression Residual Plot for V', fontsize=20)
plt.show()



