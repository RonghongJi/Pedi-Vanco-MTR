# -*- coding: utf-8 -*-
"""
Created on 2026/04/24 13:54:56

@File    :   model_dt.py
@Author  :   Ronghong Ji
"""

# %%
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np

seed = 486

# 加载数据
df = pd.read_excel(r'../data/df_rf_log_allfeature.xlsx')

# 特征列，去除 'CL' 和 'V' 列
X = df.drop(columns=['CL', 'V'])
y = df[['CL', 'V']]

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

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

# 网格搜索的参数网格
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8, None],
    'max_features': [None, 'log2', 'sqrt']
}

# 定义网格搜索模型
grid_search = GridSearchCV(DecisionTreeRegressor(random_state=seed), param_grid, scoring='r2', cv=5, n_jobs=-1)

# 进行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# 使用最佳参数训练最终模型
best_model = DecisionTreeRegressor(**best_params, random_state=seed)

# 十折交叉验证部分
kf = KFold(n_splits=10, shuffle=True, random_state=seed)

# 计算十折交叉验证的回归指标
dt_cv_metrics = cross_val_regression_metrics(best_model, X_train, y_train, kf)

# 训练最终模型并预测
best_model.fit(X_train, y_train)
dt_y_pred = best_model.predict(X_test)

# 计算测试集上的回归指标
dt_test_metrics = calculate_regression_metrics(y_test, dt_y_pred)

# 合并十折交叉验证和测试集的回归指标
metrics = {}
for col in y_train.columns:
    metrics[f'Ten_fold_CV_RMSE_{col}'] = f"{dt_cv_metrics[col]['RMSE'][0]:.2f} ± {dt_cv_metrics[col]['RMSE'][1]:.2f}"
    metrics[f'Ten_fold_CV_R2_{col}'] = f"{dt_cv_metrics[col]['R2'][0]:.2f} ± {dt_cv_metrics[col]['R2'][1]:.2f}"
    metrics[f'Ten_fold_CV_MAE_{col}'] = f"{dt_cv_metrics[col]['MAE'][0]:.2f} ± {dt_cv_metrics[col]['MAE'][1]:.2f}"

# 添加测试集指标
for col in y_test.columns:
    metrics[f'Test_RMSE_{col}'] = f"{dt_test_metrics[col]['RMSE']:.2f}"
    metrics[f'Test_R2_{col}'] = f"{dt_test_metrics[col]['R2']:.2f}"
    metrics[f'Test_MAE_{col}'] = f"{dt_test_metrics[col]['MAE']:.2f}"

# 保存到 XLSX 文件
metrics_df = pd.DataFrame(metrics, index=[0])
metrics_df.to_excel(r'../output/ML输出/DT_MultiOutput_Regression.xlsx', index=False)
metrics_df


# %%
# 保存最佳参数到 Excel 文件
best_params_df = pd.DataFrame(best_params, index=[0])
best_params_df.to_excel(r'../output/param/DT_best_params.xlsx', index=False)
best_params_df

# %%
import pandas as pd

# 如果 y_test 是 pandas DataFrame 或 Series，可以重置索引以避免问题
y_test_df = pd.DataFrame(y_test).reset_index(drop=True)  # 如果是 Series 或 DataFrame
dt_y_pred_df = pd.DataFrame(dt_y_pred).reset_index(drop=True)  # 同样处理预测值

# 确保列名称正确
y_test_df.columns = ['True Values cl', 'True Values v']
dt_y_pred_df.columns = ['Predicted Values cl', 'Predicted Values v']

# 将两者合并到一个 DataFrame 中
output_df = pd.concat([y_test_df, dt_y_pred_df], axis=1)

# 保存到 Excel 文件
output_df.to_excel(r'../output/pred/DT_predictions.xlsx', index=False)

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
    accuracies[f"Accuracy_within_{pct}%_CL"] = calculate_accuracy_within_percentage(y_test['CL'].values, dt_y_pred[:, 0], pct)
    accuracies[f"Accuracy_within_{pct}%_V"] = calculate_accuracy_within_percentage(y_test['V'].values, dt_y_pred[:, 1], pct)

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
accuracy_df.to_excel(r'../output/accuracy/DT_acc.xlsx', index=False)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 绘制 CL 拟合散点图并添加 R²
plt.figure(figsize=(10, 6))
plt.scatter(y_test['CL'], dt_y_pred[:, 0], color='black', s=15)
plt.plot([y_test['CL'].min(), y_test['CL'].max()], [y_test['CL'].min(), y_test['CL'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('DT Scatter Plot for CL', fontsize=20)

# 计算 R² 并添加到图中
r2_cl = r2_score(y_test['CL'], dt_y_pred[:, 0])
plt.text(0.8, 0.3, f"R² = {r2_cl:.2f}", fontsize=16, fontweight='bold', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.show()

# 绘制 V 拟合散点图并添加 R²
plt.figure(figsize=(10, 6))
plt.scatter(y_test['V'], dt_y_pred[:, 1], color='black', s=15)
plt.plot([y_test['V'].min(), y_test['V'].max()], [y_test['V'].min(), y_test['V'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('DT Scatter Plot for V', fontsize=20)

# 计算 R² 并添加到图中
r2_v = r2_score(y_test['V'], dt_y_pred[:, 1])
plt.text(0.8, 0.3, f"R² = {r2_v:.2f}", fontsize=16, fontweight='bold', ha='right', va='bottom', transform=plt.gca().transAxes)
plt.show()

# %%
import matplotlib.pyplot as plt
# 设置字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
# 绘制拟合散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test['CL'], dt_y_pred[:, 0], color='black', s=15)
plt.plot([y_test['CL'].min(), y_test['CL'].max()], [y_test['CL'].min(), y_test['CL'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('DT Fitted Scatter Plot for CL', fontsize=20)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test['V'], dt_y_pred[:, 1], color='black', s=15)
plt.plot([y_test['V'].min(), y_test['V'].max()], [y_test['V'].min(), y_test['V'].max()], color='black', linestyle='--')  # 45度参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Predictions', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('DT Fitted Scatter Plot for V', fontsize=20)
plt.show()

# 绘制残差图
plt.figure(figsize=(10, 6))
residuals_cl = y_test['CL'] - dt_y_pred[:, 0]
plt.scatter(y_test['CL'], residuals_cl, color='black', s=15)
plt.axhline(0, color='black', linestyle='--')  # 0参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Residuals', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('DT Residual Plot for CL', fontsize=20)
plt.show()

plt.figure(figsize=(10, 6))
residuals_v = y_test['V'] - dt_y_pred[:, 1]
plt.scatter(y_test['V'], residuals_v, color='black', s=15)
plt.axhline(0, color='black', linestyle='--')  # 0参考线
plt.xlabel('True Values', fontsize=18)
plt.ylabel('Residuals', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title('DT Residual Plot for V', fontsize=20)
plt.show()



